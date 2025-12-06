import os
import hashlib
import logging
import requests

# Framework & Utilitários
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv

# Processamento de PDF
from pypdf import PdfReader

# Rede Robusta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- CONFIGURAÇÃO INICIAL ---
load_dotenv()

# Configuração de Logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Constantes
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"

# MODELO: 'sonar-pro' (Inteligente, Lento, Detalhado)
MODEL_NAME = "sonar-pro" 

MAX_TEXT_LENGTH = 100_000  # Limite de segurança de caracteres

# Cache em Memória
RESPONSE_CACHE = {}

# --- REDE ROBUSTA ---
def get_session():
    """Cria uma sessão HTTP com estratégia de retries automática."""
    session = requests.Session()
    
    # MELHORIA: Configuração mais agressiva para erros de conexão (Socket errors)
    retry = Retry(
        total=3, 
        backoff_factor=1,  # Espera 1s, 2s, 4s (Aumentado para dar tempo à API)
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["POST"],
        connect=3, # Tenta conectar 3 vezes se der erro de socket
        read=3     # Tenta ler 3 vezes se a stream falhar
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session

http_session = get_session()

# --- UTILITÁRIOS ---
def extrair_texto_pdf(file_storage):
    """Lê PDF com tratamento de erros."""
    try:
        reader = PdfReader(file_storage)
        text_parts = []
        # Limite de páginas para evitar sobrecarga
        for i, page in enumerate(reader.pages):
            if i > 25: break 
            content = page.extract_text()
            if content:
                text_parts.append(content)
        
        full_text = "\n".join(text_parts)
        return full_text if full_text.strip() else None
    except Exception as e:
        logger.error(f"Erro ao processar PDF: {e}")
        return None

# --- ENGENHARIA DE PROMPTS (LIMITES AUMENTADOS) ---
STYLE_PROMPTS = {
    "curto": {
        "persona": "Editor Chefe de Tecnologia e Defesa do Consumidor (Estilo 'TL;DR').",
        "instruction": (
            "A tua missão é poupar tempo. Identifica IMEDIATAMENTE as 'armadilhas'. "
            "Não faças introduções. Vai direto aos factos."
            "\nESTRUTURA OBRIGATÓRIA:"
            "\n1. 💰 **Custos Reais:** (Quanto custa? Renova sozinho?)"
            "\n2. 🚨 **Riscos Críticos:** (O que perco? Onde estão os meus dados?)"
            "\n3. 🚪 **Como Sair:** (É difícil cancelar?)"
        ),
        "constraints": "Máximo 400 palavras. Usa bullet points curtos. Sem 'juridiquês'.",
        "tokens": 1500
    },
    "detalhado": {
        "persona": "Advogado Sénior Especialista em Direito do Consumidor Europeu e RGPD.",
        "instruction": (
            "Faz uma análise forense do documento. Identifica cláusulas abusivas à luz da lei portuguesa/europeia (DL 446/85, RGPD, Lei de Defesa do Consumidor). "
            "Explica o impacto prático de cada termo técnico."
            "\nESTRUTURA:"
            "\n- Análise de Privacidade (RGPD)"
            "\n- Propriedade Intelectual (Conteúdos do utilizador)"
            "\n- Resolução de Litígios (Arbitragem vs Tribunais)"
            "\n- Cláusulas de Exclusão de Responsabilidade"
        ),
        "constraints": "Cita conceitos legais relevantes. Sê exaustivo e completa o raciocínio.",
        "tokens": 8000 
    },
    "el5": {
        "persona": "Professor do Ensino Básico (Explicar a uma Criança de 10 anos).",
        "instruction": (
            "Traduz tudo para analogias do recreio ou da vida doméstica. "
            "Se fala em 'dados biométricos', diz 'o formato do teu rosto'. "
            "Se fala em 'renúncia de foro', diz 'não podes fazer queixa à professora'."
        ),
        "constraints": "Usa emojis. Linguagem super simples. Zero termos técnicos.",
        "tokens": 2000
    },
    "riscos": {
        "persona": "Auditor de Segurança Paranóico (Red Team).",
        "instruction": (
            "O teu único objetivo é encontrar motivos para NÃO ACEITAR este contrato. "
            "Ignora os benefícios. Foca-te no pior cenário possível (Worst-Case Scenario). "
            "Destaca: Venda de dados, multas escondidas, vigilância."
        ),
        "constraints": "Usa 🛑 para perigos extremos e ⚠️ para alertas. Sê alarmista mas factual.",
        "tokens": 3000
    }
}

def chamar_perplexity(texto: str, estilo_key: str, custom_prompt: str = "") -> str:
    if not PERPLEXITY_API_KEY:
        logger.critical("API Key não configurada.")
        raise RuntimeError("Erro de configuração no servidor.")

    # 1. Recuperar Configurações
    style_config = STYLE_PROMPTS.get(estilo_key, STYLE_PROMPTS["curto"])
    
    # 2. Cache Inteligente
    input_signature = f"{texto[:5000]}-{estilo_key}-{custom_prompt}"
    cache_key = hashlib.md5(input_signature.encode()).hexdigest()
    
    if cache_key in RESPONSE_CACHE:
        logger.info(f"Cache hit para: {cache_key}")
        return RESPONSE_CACHE[cache_key]

    # 3. System Prompt
    system_content = (
        "Tu és a IA 'Termos Claros'.\n"
        f"PERSONA: {style_config['persona']}\n"
        f"OBJETIVO: {style_config['instruction']}\n"
        f"RESTRIÇÕES: {style_config['constraints']}\n"
        "IDIOMA: Português de Portugal (PT-PT) nativo e fluente."
    )

    # 4. User Prompt
    user_content = (
        f"Analisa este texto legal ({len(texto)} caracteres). Texto abaixo:\n\n"
        f"'''{texto[:MAX_TEXT_LENGTH]}'''\n\n"
        "--- INSTRUÇÃO FINAL DE FORMATAÇÃO ---\n"
        "1. Começa SEMPRE com este bloco exato (usa o quote >):\n"
        "   > **⚠️ AVISO:** Análise gerada por IA (Modelo Sonar-Pro). Não dispensa consulta jurídica profissional.\n\n"
        "2. Usa Markdown rico (negrito, tabelas, listas).\n"
        "3. Se houver valores monetários ou prazos, CRIA UMA TABELA."
    )

    if custom_prompt:
        user_content += f"\n\nATENÇÃO AO PEDIDO DO UTILIZADOR: {custom_prompt}"

    # 5. Chamada API
    try:
        logger.info(f"A chamar Perplexity (Modelo: {MODEL_NAME}, Estilo: {estilo_key})...")
        
        # Preparar payload
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_content}, 
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.1, 
            "frequency_penalty": 0.5 
        }

        if style_config['tokens']:
             payload["max_tokens"] = style_config['tokens']

        response = http_session.post(
            PERPLEXITY_URL, 
            json=payload, 
            headers={
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}", 
                "Content-Type": "application/json",
                # CORREÇÃO CRÍTICA:
                "Connection": "close", # Evita 'Connection reset by peer' forçando nova conexão
                "User-Agent": "TermosClarosBot/1.0" # Identificação para evitar bloqueios de firewall
            },
            timeout=120 
        )
        response.raise_for_status()
        result = response.json()["choices"][0]["message"]["content"]
        
        # Cache
        if len(RESPONSE_CACHE) > 50:
            RESPONSE_CACHE.pop(next(iter(RESPONSE_CACHE)))
        RESPONSE_CACHE[cache_key] = result
        
        return result

    except requests.exceptions.Timeout:
        logger.error("Timeout na API da Perplexity.")
        raise RuntimeError("A IA demorou demasiado. O documento é muito extenso.")
    except Exception as e:
        logger.error(f"Erro API: {str(e)}")
        raise RuntimeError(f"Erro ao processar: {str(e)}")

# --- ROTAS ---

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/summarize", methods=["POST"])
def api_summarize():
    texto_final = ""

    # Input
    if 'file' in request.files:
        file = request.files['file']
        if file.filename != '':
            texto_final = extrair_texto_pdf(file)
            if not texto_final:
                return jsonify({"error": "PDF ilegível ou protegido."}), 400
    elif request.form.get("terms_text"):
        texto_final = request.form.get("terms_text")
    elif request.is_json:
        texto_final = request.get_json().get("terms_text", "")

    # Validação
    if not texto_final or len(texto_final.strip()) < 10:
        return jsonify({"error": "Texto insuficiente."}), 400

    estilo = request.form.get("style") or "curto"
    custom = request.form.get("custom_prompt") or ""

    try:
        resumo = chamar_perplexity(texto_final, estilo, custom)
        return jsonify({"summary": resumo})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
