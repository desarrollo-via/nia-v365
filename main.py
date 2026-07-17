"""
main.py — Endpoint principal NIA v365

Responsabilidades:
- Exponer endpoints FastAPI para chat y archivos.
- Mantener sesión conversacional en MongoDB.
- Capturar datos básicos del cliente.
- Evaluar necesidad técnica.
- Transformar mensaje natural en query limpia de catálogo.
- Buscar catálogo real.
- Validar compatibilidad producto/necesidad.
- Construir respuestas seguras cuando hay datos de catálogo.
- Evitar que el LLM invente códigos, marcas, nombres o descripciones.

Regla de arquitectura:
El LLM conversa, pero el backend decide y construye respuestas críticas
cuando hay productos reales del catálogo.
"""

import hashlib
import hmac
import logging
import logging.handlers
import os
import re
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from pydantic import BaseModel

from memory import (
    get_session,
    save_session,
    ensure_index,
    get_cliente,
    upsert_cliente,
)
from openai_client import call_nia, call_llm_json
from nia_prompt import PROMPT_MAESTRO
from catalog import (
    buscar_por_codigo,
    buscar_por_texto,
    buscar_con_campos,
    evaluar_coincidencia,
    formatear_producto,
    debe_preguntar_tipo_producto,
    extraer_campos_query,
    score_campos_producto,
    filtrar_candidatos_coherentes,
    campos_disponibles_de,
    ordenar_campos_por_prioridad,
    limpiar_envoltura_conversacional,
)
from product_matcher import validar_compatibilidad_producto
from file_processor import procesar_archivo
from knowledge import contexto_para_agente
from questions_agent import (
    generar_preguntas,
    generar_preguntas_campos_dinamicos,
)
from response_engine import (
    respuesta_producto_encontrado,
    respuesta_producto_relacionado,
    respuesta_sin_resultado,
    contiene_placeholder,
)
from bitrix_connector.router import router as bitrix_router


# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────

def setup_logging():
    """
    Configura logging para consola y archivo rotativo.

    Evita duplicar handlers cuando Uvicorn recarga con --reload.
    """
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)

    root = logging.getLogger("nia")
    root.setLevel(level)

    if root.handlers:
        return

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    fh = logging.handlers.RotatingFileHandler(
        "nia.log",
        maxBytes=10_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)

    root.addHandler(ch)
    root.addHandler(fh)


setup_logging()
logger = logging.getLogger("nia.main")


# ─────────────────────────────────────────────────────────────
# FastAPI / Rate limiting
# ─────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="NIA — Asistente Comercial ViaIndustrial")
app.state.limiter = limiter

app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(bitrix_router)


@app.on_event("startup")
async def startup():
    await ensure_index()
    logger.info("NIA arrancó correctamente")


# ─────────────────────────────────────────────────────────────
# Modelos API
# ─────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    mensaje: str
    phone_id: Optional[str] = None


class ChatResponse(BaseModel):
    respuesta: str
    etapa: Optional[str] = None
    items_resultado: Optional[list] = None
    cliente: Optional[dict] = None


# ─────────────────────────────────────────────────────────────
# WhatsApp webhook auth
# ─────────────────────────────────────────────────────────────

WA_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "")
WA_APP_SECRET = os.getenv("WHATSAPP_APP_SECRET", "")


def verificar_firma_whatsapp(request: Request, body: bytes) -> bool:
    """
    Valida la firma HMAC-SHA256 que Meta envía en X-Hub-Signature-256.

    En desarrollo, si no hay WA_APP_SECRET configurado, permite pasar.
    """
    if not WA_APP_SECRET:
        return True

    sig_header = request.headers.get("X-Hub-Signature-256", "")
    if not sig_header.startswith("sha256="):
        return False

    firma_esperada = hmac.new(
        WA_APP_SECRET.encode(),
        body,
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(sig_header[7:], firma_esperada)


@app.get("/webhook/whatsapp")
async def whatsapp_verify(
    hub_mode: str = None,
    hub_challenge: str = None,
    hub_verify_token: str = None,
):
    """
    Verificación del webhook de WhatsApp Business.
    """
    if hub_mode == "subscribe" and hub_verify_token == WA_VERIFY_TOKEN:
        logger.info("Webhook WhatsApp verificado correctamente")
        return int(hub_challenge)

    raise HTTPException(status_code=403, detail="Token inválido")


# ─────────────────────────────────────────────────────────────
# Regex / intención básica
# ─────────────────────────────────────────────────────────────

CODIGO_RE = re.compile(r"\b(\d{6})\b")
REF_RE = re.compile(r"\b(P\d{3,}|[A-Z]{1,4}\d{3,}[A-Z0-9]*)\b", re.IGNORECASE)
EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
NIT_RE = re.compile(r"\b(\d{8,10}-?\d?)\b")

# ============================================================
# CLASIFICADOR DE INTENCIÓN — COTIZACIÓN V
# ============================================================

TIPOS_MENSAJE_VALIDOS = {
    "buscar_producto",
    "instruccion_comercial",
    "dato_personal",
    "pregunta_estado",
    "cotizacion_recibida",
    "proforma_recibida",
    "link_documento",
    "saludo",
    "otro",
}

PROMPT_CLASIFICADOR_MENSAJE = """
Eres el clasificador de intención de NIA, asistente comercial técnico de ViaIndustrial.

Tu tarea es clasificar el mensaje del cliente en UNA sola categoría.

Categorías válidas:

1. "buscar_producto"
El cliente describe un producto, da código, referencia, marca, aplicación o características técnicas.
2. "instruccion_comercial"
El cliente responde al flujo comercial: confirma, niega, da cantidad, dice que cotizamos, pide continuar o cerrar.
3. "dato_personal"
El cliente entrega datos personales o comerciales: nombre, correo, empresa, NIT, RUT, teléfono.
4. "pregunta_estado"
El cliente pregunta por el estado de pedido, cotización, entrega, despacho o disponibilidad.
5. "cotizacion_recibida"
El cliente indica que ya recibió o ya tiene la cotización.
6. "proforma_recibida"
El cliente indica que ya recibió o ya tiene la proforma.
7. "link_documento"
El cliente envía un link que puede corresponder a cotización, proforma, archivo o documento.
8. "saludo"
El cliente solo saluda.
9. "otro"
No encaja claramente en las anteriores.
Responde SOLO JSON válido, sin markdown:
{
  "tipo": "categoria",
  "confianza": 0.0,
  "razon": "frase corta"
}
"""
RESPUESTAS_CORTAS_COMERCIALES = {
    "sí",
    "si",
    "no",
    "ok",
    "dale",
    "listo",
    "perfecto",
    "correcto",
    "claro",
    "bueno",
    "exacto",
    "así es",
    "asi es",
    "de acuerdo",
    "negativo",
    "está bien",
    "esta bien",
    "okay",
    "okey",
    "va",
    "por supuesto",
    "con gusto",
    "adelante",
    "procede",
    "proceder",
    "confirmado",
    "confirmo",
    "entendido",
    "recibido",
    "enterado",
    "solo esto",
    "solo eso",
    "con esto",
    "con eso",
    "eso es todo",
    "nada mas",
    "nada más",
    "es todo",
    "eso es",
    "solo",
    "eso",
}

async def clasificar_mensaje(mensaje: str, etapa: str) -> dict:
    """
    Clasifica el mensaje del cliente usando reglas rápidas + GPT.

    Diseño:
    - Primero usa reglas determinísticas para casos obvios.
    - Usa GPT solo cuando el mensaje es ambiguo.
    - Nunca deja que GPT salte guardrails comerciales.
    """
    texto = (mensaje or "").strip()
    msg_lower = texto.lower()

    if not texto:
        return {
            "tipo": "otro",
            "confianza": 0.0,
            "razon": "mensaje vacío",
        }

    # ------------------------------------------------------------
    # 1. Reglas rápidas sin GPT
    # ------------------------------------------------------------

    if re.search(r"https?://|drive\.google|dropbox|onedrive", msg_lower, re.IGNORECASE):
        return {
            "tipo": "link_documento",
            "confianza": 1.0,
            "razon": "contiene link de documento",
        }

    if any(frase in msg_lower for frase in [
        "ya tengo la cotizacion",
        "ya tengo la cotización",
        "me llego la cotizacion",
        "me llegó la cotización",
        "ya recibi la cotizacion",
        "ya recibí la cotización",
        "ya me cotizaron",
        "me enviaron la cotizacion",
        "me enviaron la cotización",
    ]):
        return {
            "tipo": "cotizacion_recibida",
            "confianza": 1.0,
            "razon": "cliente indica cotización recibida",
        }

    if any(frase in msg_lower for frase in [
        "ya tengo la proforma",
        "me llego la proforma",
        "me llegó la proforma",
        "ya recibi la proforma",
        "ya recibí la proforma",
        "me enviaron la proforma",
    ]):
        return {
            "tipo": "proforma_recibida",
            "confianza": 1.0,
            "razon": "cliente indica proforma recibida",
        }

    if re.fullmatch(r"\d{1,5}", msg_lower):
        return {
            "tipo": "instruccion_comercial",
            "confianza": 1.0,
            "razon": "cantidad numérica",
        }

    # ------------------------------------------------------------
    # Medio de pago explícito
    # ------------------------------------------------------------
    # Solo se aplica cuando NIA ya está en la etapa de pago.
    # Evita enviar PSE, transferencia o tarjeta al clasificador LLM.
    if etapa == "pago" and any(
        medio in msg_lower
        for medio in (
            "pse",
            "transferencia",
            "tarjeta",
        )
    ):
        return {
            "tipo": "instruccion_comercial",
            "confianza": 1.0,
            "razon": "medio de pago explícito",
        }

    if msg_lower.strip() in RESPUESTAS_CORTAS_COMERCIALES:
        return {
            "tipo": "instruccion_comercial",
            "confianza": 1.0,
            "razon": "respuesta corta de flujo",
        }

    if any(frase in msg_lower for frase in [
        "solo eso",
        "con eso",
        "es todo",
        "nada mas",
        "nada más",
        "cotiza",
        "coticemos",
    ]):
        return {
            "tipo": "instruccion_comercial",
            "confianza": 1.0,
            "razon": "cierre de cotización",
        }

    if re.search(r"[\w\.-]+@[\w\.-]+\.\w+", texto):
        return {
            "tipo": "dato_personal",
            "confianza": 1.0,
            "razon": "contiene correo electrónico",
        }

    if msg_lower in {"hola", "buenas", "buenos dias", "buenos días", "buen dia", "buen día"}:
        return {
            "tipo": "saludo",
            "confianza": 1.0,
            "razon": "saludo simple",
        }

    # ------------------------------------------------------------
    # 2. Clasificación GPT para casos ambiguos
    # ------------------------------------------------------------
    prompt = f"""{PROMPT_CLASIFICADOR_MENSAJE}

Etapa actual de la conversación: {etapa}
Mensaje del cliente: "{texto}"

Clasifica el mensaje.
"""

    try:
        resultado = await call_llm_json(prompt)

        tipo = str(resultado.get("tipo", "otro")).strip()
        confianza = float(resultado.get("confianza", 0.0) or 0.0)
        razon = str(resultado.get("razon", "")).strip()

        if tipo not in TIPOS_MENSAJE_VALIDOS:
            tipo = "otro"
            confianza = 0.0
            razon = "tipo inválido devuelto por clasificador"

        return {
            "tipo": tipo,
            "confianza": confianza,
            "razon": razon,
        }

    except Exception as e:
        logger.warning("clasificar_mensaje falló: %s", e)
        return {
            "tipo": "otro",
            "confianza": 0.0,
            "razon": "fallback por error del clasificador",
        }

PALABRAS_SALUDO = {"hola", "buenas", "buenos", "buen", "hi", "hello", "hey", "saludos"}
PALABRAS_MAS = {"también", "otro", "otra", "más", "adicional", "y además", "necesito más", "y también"}
PALABRAS_FIN = {"solo eso", "con eso", "es todo", "nada más", "eso es todo", "listo", "ok cotiza", "cotiza"}


def detectar_identificador(texto: str):
    """
    Detecta código exacto de 6 dígitos o referencia tipo Pxxxxx / alfanumérica.
    """
    m = CODIGO_RE.search(texto)
    if m:
        return "codigo", m.group(1)

    m = REF_RE.search(texto)
    if m:
        return "referencia", m.group(1).upper()

    return None, None


def es_solo_saludo(texto: str) -> bool:
    """
    Determina si el mensaje solo es saludo.
    """
    t = texto.lower().strip().rstrip(".,!")
    return t in PALABRAS_SALUDO or (
        len(t.split()) <= 2 and any(saludo in t for saludo in PALABRAS_SALUDO)
    )


# ─────────────────────────────────────────────────────────────
# Captura silenciosa de cliente
# ─────────────────────────────────────────────────────────────

def extraer_datos_cliente(mensaje: str, cliente_actual: dict) -> dict:
    """
    Extrae datos básicos del cliente sin interrumpir el flujo.

    Reglas:
    - Solo completa campos vacíos.
    - No sobreescribe datos ya capturados.
    - Captura email, NIT, nombre y empresa.
    - Limpia empresa para no arrastrar correo, NIT o frases posteriores.
    """
    cliente = dict(cliente_actual) if cliente_actual else {}

    if not cliente.get("email"):
        m = EMAIL_RE.search(mensaje)
        if m:
            cliente["email"] = m.group(0).strip()
            logger.debug("Email capturado: %s", cliente["email"])

    if not cliente.get("nit"):
        m = NIT_RE.search(mensaje)
        if m:
            cliente["nit"] = m.group(1).strip()
            logger.debug("NIT capturado: %s", cliente["nit"])

    if not cliente.get("nombre"):
        patrones_nombre = [
            r"(?:soy|me llamo|mi nombre es)\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){0,3})",
            r"(?:habla|llama|escribe)\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){0,2})",
        ]

        for patron in patrones_nombre:
            m = re.search(patron, mensaje, re.IGNORECASE)
            if m:
                cliente["nombre"] = m.group(1).strip()
                logger.debug("Nombre capturado: %s", cliente["nombre"])
                break

    if not cliente.get("empresa"):
        patrones_empresa = [
            r"(?:de la empresa|trabajo en|empresa|compañía)\s+(.+)",
            r"(?:somos|representamos a)\s+(.+)",
        ]

        cortes = [
            r"\s+y\s+mi\s+correo\s+es\s+",
            r"\s+y\s+mi\s+email\s+es\s+",
            r"\s+y\s+el\s+correo\s+es\s+",
            r"\s+mi\s+correo\s+es\s+",
            r"\s+mi\s+email\s+es\s+",
            r"\s+correo\s+",
            r"\s+email\s+",
            r"\s+y\s+mi\s+nit\s+es\s+",
            r"\s+mi\s+nit\s+es\s+",
            r"\s+nit\s+",
            r"\s+soy\s+",
            r"\s+me\s+llamo\s+",
            r"\s+mi\s+nombre\s+es\s+",
        ]

        for patron in patrones_empresa:
            m = re.search(patron, mensaje, re.IGNORECASE)
            if not m:
                continue

            empresa = m.group(1).strip()

            for corte in cortes:
                empresa = re.split(corte, empresa, flags=re.IGNORECASE)[0].strip()

            empresa = empresa.strip(" ,.;:-")

            if len(empresa) > 2:
                cliente["empresa"] = empresa
                logger.debug("Empresa capturada: %s", cliente["empresa"])
                break

    return cliente


# ─────────────────────────────────────────────────────────────
# Datos faltantes / saludo
# ─────────────────────────────────────────────────────────────

def datos_faltantes(cliente: dict, etapa: str) -> list:
    faltantes = []

    if etapa == "cotizacion":
        if not cliente.get("nombre"):
            faltantes.append("¿A nombre de quién va la cotización?")
        if not cliente.get("email"):
            faltantes.append("¿A qué correo te envío la cotización?")

    elif etapa == "proforma":
        if not cliente.get("empresa"):
            faltantes.append("¿Cuál es la razón social de tu empresa?")
        if not cliente.get("nit"):
            faltantes.append("¿Cuál es el NIT de la empresa?")

    return faltantes


def saludo_personalizado(cliente: dict) -> str:
    if cliente.get("nombre"):
        return f"[CLIENTE CONOCIDO: {cliente['nombre']}]"
    return "[CLIENTE NUEVO — capturar nombre si lo menciona]"


# ─────────────────────────────────────────────────────────────
# Evaluación de necesidad
# ─────────────────────────────────────────────────────────────

def _cuenta_parametros_tecnicos_generales(texto: str) -> int:
    """
    Cuenta señales técnicas generales sin depender de una familia específica.

    No es una lista de productos ni de exclusiones.
    Son patrones de magnitudes industriales comunes.
    """
    t = texto.lower()

    patrones = [
        r"\b\d+(\.\d+)?\s*(bar|psi|pa|kpa|mpa)\b",
        r"\b\d+(\.\d+)?\s*(l/min|lpm|gpm|m3/h|m³/h)\b",
        r"\b\d+(\.\d+)?\s*(°c|c|grados)\b",
        r"\b\d+(\.\d+)?\s*(v|vac|vdc|ma|a|hz|kw|hp)\b",
        r"\b\d+(\.\d+)?\s*(mm|cm|m|pulg|pulgadas|in)\b",
        r"\b(agua|aire|vapor|aceite|gas|fluido|quimico|químico)\b",
        r"\b(limpia|residual|corrosivo|industrial|sanitario|explosivo)\b",
    ]

    return sum(1 for patron in patrones if re.search(patron, t, re.IGNORECASE))


def _parece_solicitud_de_producto(texto: str) -> bool:
    """
    Detecta si el usuario está solicitando un producto o familia de producto.

    Es una detección general de intención comercial, no una regla por producto.
    """
    t = texto.lower().strip()

    patrones = [
        r"\bnecesito\b",
        r"\brequiero\b",
        r"\bbusco\b",
        r"\bcotizar\b",
        r"\bquiero\b",
        r"\bme sirve\b",
        r"\bproducto\b",
        r"\bequipo\b",
        r"\breferencia\b",
        r"\bcódigo\b",
        r"\bcodigo\b",
    ]

    return any(re.search(patron, t) for patron in patrones) and len(t.split()) >= 3


async def evaluar_necesidad(texto: str) -> dict:
    """
    Evalúa si hay suficiente información para iniciar búsqueda en catálogo.

    Principio NIA v365:
    - Si el cliente menciona una necesidad comercial clara, se permite búsqueda preliminar.
    - Buscar NO significa recomendar.
    - La recomendación final la controla product_matcher.py.
    - Si no hay match confiable, NIA pregunta más datos.
    """
    ctx = contexto_para_agente(texto)
    dominio = ctx.get("dominio", "general")

    parametros_generales = _cuenta_parametros_tecnicos_generales(texto)
    parece_solicitud = _parece_solicitud_de_producto(texto)

    if parece_solicitud:
        return {
            "clara": True,
            "preguntas": [],
            "dominio": dominio,
            "razon": "Solicitud comercial suficiente para búsqueda preliminar en catálogo.",
        }

    if parametros_generales >= 2:
        return {
            "clara": True,
            "preguntas": [],
            "dominio": dominio,
            "razon": "Solicitud con suficientes parámetros técnicos para búsqueda preliminar.",
        }

    prompt_evaluacion = (
        f"El cliente de una empresa de instrumentación industrial dice: \"{texto}\"\n\n"
        f"Dominio detectado: {dominio}\n\n"
        "¿Hay suficiente información para iniciar una búsqueda preliminar en catálogo?\n"
        "No significa recomendar todavía; solo buscar candidatos reales.\n"
        "Responde SOLO JSON sin markdown:\n"
        "{\"clara\": true/false, \"razon\": \"una frase corta\"}"
    )

    try:
        resultado = await call_llm_json(prompt_evaluacion)
        clara = bool(resultado.get("clara", False))

        logger.debug(
            "LLM evalúa necesidad: clara=%s razón=%s",
            clara,
            resultado.get("razon", ""),
        )

        if clara:
            return {
                "clara": True,
                "preguntas": [],
                "dominio": dominio,
                "razon": resultado.get("razon", ""),
            }

    except Exception as exc:
        logger.warning("LLM evaluación fallida, usando preguntas: %s", exc)

    preguntas = await generar_preguntas(texto)

    return {
        "clara": False,
        "preguntas": preguntas,
        "dominio": dominio,
    }

# ============================================================
# MATCH TEXTUAL SEGURO SOBRE CANDIDATO DE CATÁLOGO
# ============================================================

PALABRAS_FUNCIONALES_MATCH = {
    "necesito",
    "necesitamos",
    "quiero",
    "requiero",
    "requiere",
    "busco",
    "buscar",
    "cotizar",
    "cotizacion",
    "cotización",
    "producto",
    "equipo",
    "sistema",
    "para",
    "con",
    "una",
    "uno",
    "unos",
    "unas",
    "del",
    "de",
    "la",
    "el",
    "los",
    "las",
    "que",
    "por",
    "favor",
}


def _normalizar_match_textual(valor: str) -> str:
    """
    Normaliza texto para comparar intención del cliente contra nombre/descr.
    No decide compatibilidad técnica; solo ayuda a detectar coincidencias claras.
    """
    if not valor:
        return ""

    valor = str(valor).lower()
    valor = re.sub(r"[^\w\sáéíóúñü-]", " ", valor, flags=re.IGNORECASE)
    valor = re.sub(r"\s+", " ", valor).strip()

    # Normalización simple de tildes sin depender de librerías externas.
    reemplazos = {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "ü": "u",
        "ñ": "n",
    }

    for origen, destino in reemplazos.items():
        valor = valor.replace(origen, destino)

    return valor


def _tokens_producto_cliente(texto: str) -> list[str]:
    """
    Extrae tokens útiles de una necesidad del cliente.

    Importante:
    - No es una lista de productos.
    - Solo elimina palabras funcionales del lenguaje.
    """
    texto_norm = _normalizar_match_textual(texto)

    return [
        token
        for token in texto_norm.split()
        if len(token) > 2
        and not token.isdigit()
        and token not in PALABRAS_FUNCIONALES_MATCH
    ]


def _cobertura_tokens_en_texto(tokens: list[str], texto_objetivo: str) -> float:
    """
    Calcula cuántos tokens de la solicitud aparecen en el texto del producto.
    """
    if not tokens:
        return 0.0

    objetivo_norm = _normalizar_match_textual(texto_objetivo)
    objetivo_tokens = set(objetivo_norm.split())

    if not objetivo_tokens:
        return 0.0

    encontrados = 0

    for token in tokens:
        if token in objetivo_tokens:
            encontrados += 1
            continue

        # Coincidencia flexible para singular/plural o variantes pequeñas.
        if any(token in obj or obj in token for obj in objetivo_tokens if len(obj) > 3):
            encontrados += 1

    return encontrados / len(tokens)


def _debe_promover_related_a_exacto(
    texto_cliente: str,
    producto: dict,
    estado_match: str,
    campos_query: Optional[dict] = None,
) -> bool:
    """
    Promueve un related_match a exact_match solo cuando hay evidencia textual fuerte.

    Regla segura:
    - Solo aplica si el matcher ya encontró un producto relacionado.
    - No aplica cuando hay campos técnicos detectados, porque allí conviene validar.
    - Requiere mínimo 2 tokens útiles.
    - Requiere alta cobertura en nombre o descripción corta.

    Esto evita parches específicos como:
    if "bomba centrifuga" in texto
    """
    if estado_match != "related_match":
        return False

    if not producto:
        return False

    # Si hay campos técnicos, preferimos mantener validación técnica.
    # Ejemplo: rango, salida, conexión, voltaje, material.
    if campos_query:
        return False

    tokens = _tokens_producto_cliente(texto_cliente)

    if len(tokens) < 2:
        return False

    nombre = producto.get("nombre") or ""
    descripcion_corta = producto.get("descripcion_corta") or ""
    categoria = " ".join(
        [
            str(producto.get("categoria") or ""),
            str(producto.get("nivel_3") or ""),
            str(producto.get("nivel_4") or ""),
        ]
    )

    cobertura_nombre = _cobertura_tokens_en_texto(tokens, nombre)
    cobertura_desc = _cobertura_tokens_en_texto(tokens, descripcion_corta)
    cobertura_categoria = _cobertura_tokens_en_texto(tokens, categoria)

    logger.info(
        "Evaluando promocion related->exact tokens=%s codigo=%s cobertura_nombre=%.2f cobertura_desc=%.2f cobertura_categoria=%.2f",
        tokens,
        producto.get("codigo"),
        cobertura_nombre,
        cobertura_desc,
        cobertura_categoria,
    )

    # Caso fuerte:
    # "bomba centrifuga" dentro de "Bomba centrifuga autocebante"
    if cobertura_nombre >= 0.90:
        return True

    # Caso aceptable si nombre + descripción/categoría respaldan la misma intención.
    if cobertura_nombre >= 0.75 and (cobertura_desc >= 0.75 or cobertura_categoria >= 0.75):
        return True

    return False

# ─────────────────────────────────────────────────────────────
# Búsqueda catálogo / compatibilidad
# ─────────────────────────────────────────────────────────────

async def generar_queries_catalogo(texto: str) -> list[str]:
    """
    Convierte el mensaje natural del cliente en una o varias consultas útiles
    para el catálogo.

    Responsabilidad:
    - Separar conversación humana de búsqueda técnica.
    - No modificar catalog.py para entender frases conversacionales.
    - No usar listas rígidas de palabras prohibidas.
    - Pedir al LLM una frase compacta de búsqueda, pero con fallback seguro.
    """
    texto = (texto or "").strip()

    if not texto:
        return []

    queries = []

    prompt = f"""
Eres un normalizador de búsqueda para un catálogo industrial.

Convierte el mensaje del cliente en una consulta corta para buscar productos reales en catálogo.

REGLAS:
1. No inventes productos.
2. No agregues marcas, referencias ni datos que el cliente no dijo.
3. Conserva el tipo de producto solicitado.
4. Conserva atributos técnicos relevantes: presión, caudal, señal, voltaje, material, fluido, referencia, marca si existen.
5. Quita intención conversacional como "necesito", "busco", "quiero cotizar", pero sin perder el producto.
6. Devuelve SOLO JSON válido.

Mensaje del cliente:
{texto}

Formato:
{{
  "queries": ["consulta principal", "consulta alternativa opcional"]
}}
"""

    try:
        data = await call_llm_json(prompt)
        raw_queries = data.get("queries", [])

        if isinstance(raw_queries, list):
            for query in raw_queries:
                q = str(query).strip()
                if q and q not in queries:
                    queries.append(q)

    except Exception as exc:
        logger.warning("No se pudo generar query limpia con LLM: %s", exc)

    if texto not in queries:
        queries.append(texto)

    return queries[:3]


async def buscar_en_catalogo(texto: str) -> dict:
    """
    Busca candidatos reales y valida compatibilidad producto/necesidad.

    Flujo correcto:
    1. Recibe mensaje natural del cliente.
    2. Genera queries limpias para catálogo.
    3. Busca candidatos reales en MongoDB.
    4. Valida compatibilidad contra la necesidad original.
    5. Retorna encontrado / relacionado / sin_resultado.
    """
    logger.info("Búsqueda catálogo solicitada: '%s'", texto[:100])

    texto_matching = (
        limpiar_envoltura_conversacional(texto)
        or str(texto or "").strip()
    )

    logger.info(
        "Texto matching: original='%s' validacion='%s'",
        texto,
        texto_matching,
    )

    campos_originales = extraer_campos_query(texto)

    campos_originales_significativos = {
        nombre: valor
        for nombre, valor in (campos_originales or {}).items()
        if (
            nombre != "valor_tecnico"
            and str(valor or "").strip()
        )
    }

    queries_generadas = await generar_queries_catalogo(texto)

    # ------------------------------------------------------------
    # Orden de consultas
    # ------------------------------------------------------------
    # Cuando el cliente ya declaró dos o más campos técnicos,
    # probamos primero el mensaje original.
    queries_catalogo = []

    if len(campos_originales_significativos) >= 2:
        texto_original_limpio = str(texto or "").strip()

        if texto_original_limpio:
            queries_catalogo.append(
                texto_original_limpio
            )

    for query_generada in queries_generadas or []:
        query_generada = str(
            query_generada or ""
        ).strip()

        if not query_generada:
            continue

        if query_generada in queries_catalogo:
            continue

        queries_catalogo.append(
            query_generada
        )

    logger.info(
        "Queries catálogo ordenadas: originales=%s queries=%s",
        campos_originales_significativos,
        queries_catalogo,
    )

    if not queries_catalogo:
        return {
            "estado": "sin_resultado",
            "razon": "No se pudo construir una consulta válida para catálogo.",
            "pregunta_sugerida": "¿Puedes indicar el tipo de producto o referencia que necesitas?",
            "candidatos_encontrados": False,
        }

    resultados = None
    query_usada = None
    campos_query = {}

    for query in queries_catalogo:
        logger.info("Intentando búsqueda catálogo con query limpia: '%s'", query)

        # Búsqueda híbrida:
        # - Limpia lenguaje conversacional.
        # - Extrae campos técnicos si existen.
        # - Consulta MongoDB como fuente oficial.
        # - NO decide compatibilidad final.
        resultados_intento, campos_detectados_intento = (
            await buscar_con_campos(query)
        )

        if resultados_intento:
            resultados = resultados_intento

            # Los campos detectados en la consulta utilizada ayudan
            # al retrieval, pero los campos del mensaje original
            # tienen prioridad semántica.
            campos_query = {
                **(campos_detectados_intento or {}),
                **campos_originales_significativos,
            }

            query_usada = query
            logger.info(
                "Catálogo devolvió %s candidatos usando query='%s' campos_query=%s",
                len(resultados),
                query,
                campos_query,
            )
            break

    if not resultados:
        return {
            "estado": "sin_resultado",
            "razon": "No se encontraron candidatos reales en catálogo.",
            "pregunta_sugerida": "¿Puedes darme una referencia, marca, aplicación exacta o especificación adicional?",
            "candidatos_encontrados": False,
        }

    ok_textual, prod_textual = evaluar_coincidencia(
        resultados,
        texto_matching,
        campos=len(campos_query) if campos_query else 1,
        campos_query=campos_query,
    )

    if ok_textual and prod_textual:
        logger.info(
            "Mejor candidato textual: %s score=%s query='%s'",
            prod_textual.get("codigo"),
            prod_textual.get("_score"),
            query_usada,
        )

    # ------------------------------------------------------------
    # Guardrail determinístico de compatibilidad estructurada
    # ------------------------------------------------------------
    # El mensaje original es la fuente de verdad técnica.
    #
    # Si el cliente declaró dos o más campos, el scorer determinístico
    # encontró un candidato válido y ese candidato cumple prácticamente
    # todos los campos, no permitimos que el LLM lo sustituya por un
    # producto técnicamente inferior.
    #
    # Ejemplo validado:
    # - solicitado: rango -50 a 200 °C
    # - solicitado: sonda 150 mm x 4 mm
    # - 130617: cumple ambos campos
    # - 130620: cumple rango, pero la sonda mide 63,5 mm
    cantidad_campos_originales = len(
        campos_originales_significativos
    )

    score_estructurado_determinista = 0.0

    if (
        ok_textual
        and prod_textual
        and cantidad_campos_originales >= 2
    ):
        score_estructurado_determinista = (
            score_campos_producto(
                producto=prod_textual,
                campos_query=campos_query,
            )
        )

        logger.info(
            "Guardrail técnico determinístico: codigo=%s "
            "score_textual=%s score_estructurado=%s campos=%s",
            prod_textual.get("codigo"),
            prod_textual.get("_score"),
            score_estructurado_determinista,
            campos_query,
        )

    if (
        ok_textual
        and prod_textual
        and cantidad_campos_originales >= 2
        and score_estructurado_determinista >= 0.90
    ):
        razon_determinista = (
            "El producto coincide con la familia solicitada y cumple "
            "los campos técnicos estructurados declarados por el cliente."
        )

        prod_textual["_compatibilidad"] = {
            "estado": "exact_match",
            "confianza": score_estructurado_determinista,
            "razon": razon_determinista,
            "query_catalogo": query_usada,
            "origen": "guardrail_deterministico",
        }

        logger.info(
            "Candidato determinístico aprobado antes del product_matcher: "
            "codigo=%s score_estructurado=%s",
            prod_textual.get("codigo"),
            score_estructurado_determinista,
        )

        return {
            "estado": "encontrado",
            "producto": prod_textual,
            "tipo": "compatibilidad_estructurada",
            "exacto": True,
            "razon": razon_determinista,
            "query_catalogo": query_usada,
            "candidatos_encontrados": True,
        }

    # Si el candidato determinístico no alcanza la evidencia requerida,
    # conservamos el product_matcher como segunda capa de validación.
    decision = await validar_compatibilidad_producto(
        necesidad_cliente=texto_matching,
        candidatos=resultados,
        contexto_tecnico={
            "query_catalogo": query_usada,
            "queries_intentadas": queries_catalogo,
            "campos_query": campos_query,
        },
    )

    estado_match = decision.get("estado")
    producto = decision.get("producto")

    # ------------------------------------------------------------
    # Promoción segura related_match -> exact_match
    # ------------------------------------------------------------
    # El product_matcher puede ser conservador y marcar como relacionado
    # un producto que textualmente coincide muy bien con la necesidad.
    #
    # No bajamos umbrales globales.
    # No quemamos productos.
    # No aplica si hay campos técnicos, porque ahí sí conviene validar.
    if _debe_promover_related_a_exacto(
        texto_cliente=texto_matching,
        producto=producto,
        estado_match=estado_match,
        campos_query=campos_query,
    ):
        logger.info(
            "Promoviendo related_match a exact_match por coincidencia textual fuerte codigo=%s",
            producto.get("codigo") if producto else None,
        )

        estado_match = "exact_match"
        decision["estado"] = "exact_match"
        decision["razon"] = (
            decision.get("razon")
            or "El nombre del producto coincide claramente con la necesidad indicada."
        )

    if estado_match == "exact_match" and producto:
        producto["_compatibilidad"] = {
            "estado": "exact_match",
            "confianza": decision.get("confianza"),
            "razon": decision.get("razon"),
            "query_catalogo": query_usada,
        }

        return {
            "estado": "encontrado",
            "producto": producto,
            "tipo": "compatibilidad_exacta",
            "exacto": True,
            "razon": decision.get("razon"),
            "query_catalogo": query_usada,
            "candidatos_encontrados": True,
        }

    if estado_match == "related_match" and producto:
        producto["_compatibilidad"] = {
            "estado": "related_match",
            "confianza": decision.get("confianza"),
            "razon": decision.get("razon"),
            "query_catalogo": query_usada,
        }

        # Si el producto es relacionado, NIA no debe tomarlo como solución exacta.
        # Generamos hasta 3 preguntas técnicas para validar compatibilidad real
        # usando questions_agent.py + product_fields.py.
        preguntas_tecnicas = []

        try:
            preguntas_tecnicas = await generar_preguntas(texto)
        except Exception as e:
            logger.warning(
                "No fue posible generar preguntas técnicas para producto relacionado: %s",
                e,
            )
            preguntas_tecnicas = []

        preguntas_limpias = [
            p.strip()
            for p in preguntas_tecnicas
            if isinstance(p, str) and p.strip()
        ][:3]

        return {
            "estado": "relacionado",
            "producto": producto,
            "tipo": "producto_relacionado",
            "exacto": False,
            "razon": decision.get("razon"),
            "pregunta_sugerida": decision.get("pregunta_sugerida"),
            "query_catalogo": query_usada,
            "candidatos_encontrados": True,
            "texto_original": texto,
            "preguntas_tecnicas": preguntas_limpias,
        }

    return {
        "estado": "sin_resultado",
        "razon": decision.get("razon"),
        "pregunta_sugerida": decision.get("pregunta_sugerida"),
        "query_catalogo": query_usada,
        "candidatos_encontrados": True,
    }


async def rama_codigo(valor: str, tipo: str) -> dict:
    """
    Rama de búsqueda exacta por código o referencia.
    """
    logger.info("Búsqueda exacta: %s=%s", tipo, valor)

    prod = await buscar_por_codigo(valor)

    if prod:
        return {
            "estado": "encontrado",
            "producto": prod,
            "tipo": tipo,
            "exacto": True,
            "candidatos_encontrados": True,
        }

    logger.info("Fallback catálogo para identificador: %s", valor)

    res = await buscar_en_catalogo(valor)

    if res["estado"] in {"encontrado", "relacionado"}:
        res["tipo"] = "fallback_identificador"
        return res

    logger.info("Sin resultado para identificador: %s", valor)

    return {
        "estado": "sin_resultado",
        "tipo": tipo,
        "pregunta_sugerida": "¿Puedes verificar el código o compartir marca/referencia adicional?",
        "candidatos_encontrados": res.get("candidatos_encontrados", False),
    }


def debe_intentar_enriquecimiento(res: dict) -> bool:
    """
    Decide si vale la pena intentar Libros Rol 2.

    Regla:
    - Si no hubo candidatos en catálogo, sí tiene sentido enriquecer la búsqueda.
    - Si sí hubo candidatos, pero product_matcher dijo que ninguno es compatible,
      NO se debe enriquecer a ciegas. Se debe responder seguro y pedir precisión.
    """
    if res.get("estado") != "sin_resultado":
        return False

    return res.get("candidatos_encontrados") is False


async def enriquecer_y_buscar(texto: str) -> dict:
    """
    Usa contexto de conocimiento para enriquecer la búsqueda,
    pero mantiene la validación de compatibilidad.
    """
    ctx = contexto_para_agente(texto)
    terminos = ctx.get("terminos", [])
    dominio = ctx.get("dominio", "")
    query = f"{texto} {' '.join(terminos[:4])}".strip()

    logger.info("Libros Rol 2 — query enriquecida: '%s'", query[:80])

    res = await buscar_en_catalogo(query)

    if res["estado"] in {"encontrado", "relacionado"}:
        res["tipo"] = "libros_rol2"
        res["dominio"] = dominio
        res["query_enriquecida"] = query
        return res

    if res.get("candidatos_encontrados") is True:
        res["dominio"] = dominio
        res["query_enriquecida"] = query
        return res

    preguntas = await generar_preguntas(texto)

    return {
        "estado": "pendiente",
        "preguntas": preguntas,
        "dominio": dominio,
    }


# ─────────────────────────────────────────────────────────────
# Response helpers
# ─────────────────────────────────────────────────────────────

def _marcar_respuesta_segura(texto: str) -> str:
    """
    Marca una respuesta para que no sea reescrita por el LLM.
    """
    return "[RESPUESTA_SEGURA]\n" + texto

def _limpiar_preguntas_tecnicas(preguntas: list) -> list[str]:
    """
    Limpia y limita preguntas técnicas a máximo 3.

    - NIA puede tener hasta 3 preguntas técnicas.
    - Pero debe hacer una sola pregunta por mensaje.
    """
    preguntas_limpias = []

    for pregunta in preguntas or []:
        if not isinstance(pregunta, str):
            continue

        p = pregunta.strip()

        if not p:
            continue

        # Limpieza defensiva por si el LLM devuelve numeración.
        p = re.sub(r"^\s*(\d+[\.\)]|[-*•])\s*", "", p).strip()

        if p:
            preguntas_limpias.append(p)

    return preguntas_limpias[:3]


def _crear_ctx_preguntas_secuenciales(
    texto_original: str,
    preguntas: list[str],
    necesidad_ctx_base: Optional[dict] = None,
) -> dict:
    """
    Crea el estado conversacional para hacer preguntas técnicas una por una.

    Estructura:
    - texto_original: necesidad inicial del cliente.
    - preguntas_tecnicas: lista completa de preguntas.
    - pregunta_actual_idx: índice de la pregunta que NIA acaba de enviar.
    - respuestas_tecnicas: respuestas que va dando el cliente.
    """
    necesidad_ctx_base = necesidad_ctx_base or {}

    return {
        **necesidad_ctx_base,
        "texto_original": texto_original,
        "preguntas_tecnicas": preguntas,
        "pregunta_actual_idx": 0,
        "respuestas_tecnicas": [],
    }


def _respuesta_pregunta_tecnica_unica(
    pregunta: str,
    primera: bool = False,
) -> str:
    """
    Construye una respuesta segura con UNA sola pregunta técnica.

    No enumera 3 preguntas.
    No muestra bloques largos.
    Mantiene conversación natural.
    """
    pregunta = (pregunta or "").strip()

    if primera:
        return (
            "Entiendo. Para ayudarte mejor, necesito confirmar primero este dato:\n\n"
            f"{pregunta}"
        )

    return (
        "Gracias. Ahora necesito confirmar lo siguiente:\n\n"
        f"{pregunta}"
    )

def _crear_ctx_tipo_producto(
    texto_original: str,
    tipos_detectados: list[str],
) -> dict:
    """
    Crea el contexto conversacional para cuando NIA necesita confirmar
    primero el tipo/familia de producto.

    Este estado es diferente a preguntas_tecnicas:
    - tipo_producto: pregunta por rama del catálogo.
    - preguntas_tecnicas: pregunta por especificaciones de aplicación.
    """
    tipos_limpios = [
        str(t).strip()
        for t in tipos_detectados or []
        if str(t).strip()
    ][:5]

    return {
        "texto_original": texto_original,
        "tipos_detectados": tipos_limpios,
        "tipo_confirmado": None,
    }


def _respuesta_pregunta_tipo_producto(
    texto_original: str,
    tipos_detectados: list[str],
) -> str:
    """
    Construye una pregunta natural para confirmar tipo de producto.

    Regla Don Andrés:
    - No inventar tipos.
    - Solo mostrar opciones detectadas desde catálogo real.
    - Hacer una sola pregunta.
    - Mostrar opciones de forma limpia y legible.
    """
    tipos_limpios = []

    for tipo in tipos_detectados or []:
        tipo_limpio = str(tipo).strip()

        if not tipo_limpio:
            continue

        # Limpieza visual defensiva.
        # No cambia la lógica del catálogo; solo mejora cómo se muestra.
        tipo_limpio = re.sub(r"\s+", " ", tipo_limpio).strip()

        # Correcciones visuales comunes por textos taxonómicos pegados.
        tipo_limpio = tipo_limpio.replace("portatilesbolsillo", "portatiles bolsillo")
        tipo_limpio = tipo_limpio.replace("digitalesportatiles", "digitales portatiles")

        if tipo_limpio not in tipos_limpios:
            tipos_limpios.append(tipo_limpio)

        if len(tipos_limpios) >= 5:
            break

    if not tipos_limpios:
        return (
            "Para ayudarte mejor, necesito confirmar primero el tipo de producto. "
            "¿Qué variante o aplicación específica necesitas?"
        )

    if len(tipos_limpios) == 1:
        opciones = tipos_limpios[0]
    elif len(tipos_limpios) == 2:
        opciones = f"{tipos_limpios[0]} o {tipos_limpios[1]}"
    else:
        opciones = ", ".join(tipos_limpios[:-1]) + f" o {tipos_limpios[-1]}"

    producto_base = (texto_original or "producto").strip()
    producto_base = re.sub(r"\s+", " ", producto_base)

    return (
        f"Encontré varias líneas relacionadas con {producto_base} en el catálogo. "
        f"¿Qué tipo necesitas: {opciones}?"
    )

def construir_respuesta_desde_resultado(
    res: dict,
    cliente: dict,
    productos_acumulados: list,
    desde: str,
    necesidad_ctx_base: Optional[dict] = None,
) -> tuple[str, str, dict]:
    """
    Convierte un resultado de catálogo en:
    - contexto_extra
    - nueva_etapa
    - necesidad_ctx actualizado

    Reglas:
    - encontrado: se agrega al carrito.
    - relacionado: no se agrega al carrito; se pide confirmación.
    - pendiente/sin_resultado/no_compatible: se mantiene descubrimiento.
    - nunca debe retornar None.
    """
    necesidad_ctx_base = necesidad_ctx_base or {}

    # Defensive: si llega algo no dict, no rompemos el flujo.
    if not isinstance(res, dict):
        return (
            _marcar_respuesta_segura(
                respuesta_sin_resultado(
                    pregunta_sugerida=(
                        "¿Puedes indicarme una referencia, aplicación exacta "
                        "o especificación técnica adicional?"
                    ),
                    cliente=cliente,
                )
            ),
            "descubrimiento",
            necesidad_ctx_base,
        )

    estado = res.get("estado")

    # ------------------------------------------------------------
    # 1) Producto encontrado
    # ------------------------------------------------------------
    if estado == "encontrado" and res.get("producto"):
        producto = res["producto"]

        productos_acumulados.append({
            "producto": producto,
            "cantidad": None,
            "desde": desde,
            "ts": datetime.utcnow().isoformat(),
        })

        return (
            _marcar_respuesta_segura(
                respuesta_producto_encontrado(producto, cliente)
            ),
            "producto_encontrado",
            {},
        )

    # ------------------------------------------------------------
    # 2) Producto relacionado
    # ------------------------------------------------------------
    if estado == "relacionado" and res.get("producto"):
        producto = res["producto"]

        preguntas_limpias = _limpiar_preguntas_tecnicas(
            res.get("preguntas_tecnicas") or []
        )

        diagnostico_tecnico_completo = bool(
            necesidad_ctx_base.get(
                "diagnostico_tecnico_completo",
                False,
            )
        )

        # Si ya se completó el máximo de preguntas técnicas,
        # no se permite iniciar una lista nueva.
        #
        # En ese caso el flujo continúa hacia validando_relacionado,
        # mostrando el candidato real y pidiendo una confirmación,
        # pero sin hacer una cuarta pregunta técnica.
        if preguntas_limpias and not diagnostico_tecnico_completo:
            texto_original = (
                necesidad_ctx_base.get("texto_original")
                or necesidad_ctx_base.get("query_evaluada")
                or ""
            )

            necesidad_ctx = _crear_ctx_preguntas_secuenciales(
                texto_original=texto_original,
                preguntas=preguntas_limpias,
                necesidad_ctx_base={
                    **necesidad_ctx_base,
                    "producto_relacionado": producto,
                    "pregunta_sugerida": res.get("pregunta_sugerida"),
                    "razon": res.get("razon"),
                },
            )

            respuesta = _respuesta_pregunta_tecnica_unica(
                preguntas_limpias[0],
                primera=True,
            )

            return (
                _marcar_respuesta_segura(respuesta),
                "descubrimiento",
                necesidad_ctx,
            )

        # Si no hay preguntas técnicas, pedimos confirmación del relacionado.
        respuesta = respuesta_producto_relacionado(
            producto=producto,
            razon=res.get("razon"),
            pregunta_sugerida=res.get("pregunta_sugerida"),
            cliente=cliente,
        )

        necesidad_ctx = {
            **necesidad_ctx_base,
            "producto_relacionado": producto,
            "pregunta_sugerida": res.get("pregunta_sugerida"),
            "razon": res.get("razon"),
        }

        return (
            _marcar_respuesta_segura(respuesta),
            "validando_relacionado",
            necesidad_ctx,
        )

    # ------------------------------------------------------------
    # 3) Pendiente: necesita preguntas técnicas
    # ------------------------------------------------------------
    if estado == "pendiente":
        preguntas = _limpiar_preguntas_tecnicas(res.get("preguntas", []))

        if preguntas:
            texto_original = (
                necesidad_ctx_base.get("texto_original")
                or necesidad_ctx_base.get("query_evaluada")
                or ""
            )

            necesidad_ctx = _crear_ctx_preguntas_secuenciales(
                texto_original=texto_original,
                preguntas=preguntas,
                necesidad_ctx_base=necesidad_ctx_base,
            )

            return (
                _marcar_respuesta_segura(
                    _respuesta_pregunta_tecnica_unica(
                        preguntas[0],
                        primera=True,
                    )
                ),
                "descubrimiento",
                necesidad_ctx,
            )

        return (
            _marcar_respuesta_segura(
                "Necesito un poco más de información para buscar la opción adecuada. "
                "¿Puedes indicarme la aplicación o especificación principal?"
            ),
            "descubrimiento",
            necesidad_ctx_base,
        )

    # ------------------------------------------------------------
    # 4) Fallback defensivo final
    # ------------------------------------------------------------
    # Puede llegar aquí cuando buscar_en_catalogo() retorna estados como:
    # - sin_resultado
    # - no_compatible
    # - sin_match
    # - cualquier estado futuro no contemplado
    #
    # En todos esos casos, NIA debe pedir más información sin romper el turno.
    pregunta_fallback = (
        res.get("pregunta_sugerida")
        or "¿Puedes indicarme una referencia, aplicación exacta o especificación técnica adicional?"
    )

    return (
        _marcar_respuesta_segura(
            respuesta_sin_resultado(
                pregunta_sugerida=pregunta_fallback,
                cliente=cliente,
            )
        ),
        "descubrimiento",
        necesidad_ctx_base,
    )


def _extraer_respuesta_segura(contexto_extra: str) -> Optional[str]:
    """
    Extrae respuesta segura marcada.
    """
    if contexto_extra.startswith("[RESPUESTA_SEGURA]"):
        return contexto_extra.replace("[RESPUESTA_SEGURA]\n", "", 1).strip()

    return None

# ─────────────────────────────────────────────────────────────
# Estado comercial prioritario
# ─────────────────────────────────────────────────────────────

def _normalizar_intencion(texto: str) -> str:
    """
    Normaliza texto corto para interpretar confirmaciones,
    cierres y respuestas comerciales simples.

    No se usa para catálogo. Solo para control de flujo.
    """
    t = (texto or "").lower().strip()
    reemplazos = {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "ñ": "n",
    }

    for origen, destino in reemplazos.items():
        t = t.replace(origen, destino)

    t = re.sub(r"\s+", " ", t)
    return t.strip(" .,!¡¿?")


def _es_confirmacion_afirmativa(texto: str) -> bool:
    """
    Detecta respuestas afirmativas del cliente.

    Aplica para confirmar producto sugerido, no para buscar catálogo.
    """
    t = _normalizar_intencion(texto)

    afirmaciones_exactas = {
        "si",
        "correcto",
        "ese",
        "esa",
        "ese me sirve",
        "esa me sirve",
        "me sirve",
        "sirve",
        "ok",
        "dale",
        "perfecto",
        "confirmo",
        "confirmado",
        "listo",
        "esta bien",
    }

    return t in afirmaciones_exactas


def _es_confirmacion_negativa(texto: str) -> bool:
    """
    Detecta rechazo del producto sugerido.
    """
    t = _normalizar_intencion(texto)

    negativas_exactas = {
        "no",
        "no me sirve",
        "no es",
        "no corresponde",
        "otro",
        "otra",
        "diferente",
    }

    return t in negativas_exactas


def _extraer_cantidad_solicitada(texto: str) -> Optional[int]:
    """
    Extrae una cantidad comercial cuando NIA está esperando cantidad.

    Regla:
    - Acepta cantidades razonables de 1 a 5 dígitos.
    - No interpreta números largos como cantidad para evitar confundir NIT,
      teléfonos o códigos.
    """
    if not texto:
        return None

    t = texto.lower().strip()

    m = re.search(
        r"\b(\d{1,5})\b\s*(und|unds|unidad|unidades|pieza|piezas|u)?\b",
        t,
        re.IGNORECASE,
    )

    if not m:
        return None

    try:
        cantidad = int(m.group(1))
    except ValueError:
        return None

    if cantidad <= 0:
        return None

    return cantidad


def _asignar_cantidad_ultimo_producto(productos_acumulados: list, cantidad: int) -> None:
    """
    Asigna la cantidad al último producto acumulado que aún no tenga cantidad.
    Si todos tienen cantidad, actualiza el último producto como decisión comercial.
    """
    if not productos_acumulados:
        return

    for item in reversed(productos_acumulados):
        if not item.get("cantidad"):
            item["cantidad"] = cantidad
            return

    productos_acumulados[-1]["cantidad"] = cantidad


def _parece_nombre_simple(texto: str) -> Optional[str]:
    """
    Captura nombres escritos de forma directa, por ejemplo:
    - Luis
    - Luis Díaz
    - Juan Carlos Pérez

    No captura correos, números, NIT, frases de cierre ni solicitudes de producto.
    """
    if not texto:
        return None

    limpio = texto.strip(" ,.;:-")

    if not limpio or EMAIL_RE.search(limpio) or NIT_RE.search(limpio):
        return None

    t = _normalizar_intencion(limpio)

    # ------------------------------------------------------------
    # Guardrail: respuestas cortas comerciales no son nombres
    # ------------------------------------------------------------
    #
    # RESPUESTAS_CORTAS_COMERCIALES ya es la fuente central para
    # estas expresiones, por lo que evitamos mantener otra lista
    # incompleta dentro del extractor de nombres.
    if (
        t in RESPUESTAS_CORTAS_COMERCIALES
        or t in PALABRAS_FIN
        or _es_confirmacion_afirmativa(limpio)
        or _es_confirmacion_negativa(limpio)
    ):
        return None

    bloqueados = PALABRAS_SALUDO | PALABRAS_FIN | {
        "si",
        "no",
        "ok",
        "dale",
        "correcto",
        "solo",
        "eso",
        "solo eso",
        "cotiza",
    }

    if t in bloqueados:
        return None

    if _parece_solicitud_de_producto(limpio):
        return None

    partes = limpio.split()

    if not (1 <= len(partes) <= 4):
        return None

    patron_nombre = re.compile(r"^[A-Za-zÁÉÍÓÚáéíóúÑñüÜ'-]{2,}$")

    if not all(patron_nombre.match(p) for p in partes):
        return None

    return " ".join(p.capitalize() for p in partes)


def _parece_empresa_simple(texto: str) -> Optional[str]:
    """
    Captura razón social escrita de forma directa cuando NIA está en etapa proforma.

    Ejemplos:
    - ViaIndustrial SAS
    - Equipos Industriales Fenix S.A.S
    - Industrias ABC
    """
    if not texto:
        return None

    limpio = texto.strip(" ,.;:-")

    if not limpio or EMAIL_RE.search(limpio) or NIT_RE.search(limpio):
        return None

    if _es_confirmacion_afirmativa(limpio) or _es_confirmacion_negativa(limpio):
        return None

    if _parece_solicitud_de_producto(limpio):
        return None

    if len(limpio) < 3 or len(limpio) > 80:
        return None

    if not re.search(r"[A-Za-zÁÉÍÓÚáéíóúÑñ]", limpio):
        return None

    return limpio

def _extraer_datos_contacto_desde_mensaje(mensaje: str) -> dict:

    datos = {}
    if not mensaje:
        return datos

    texto_original = str(mensaje).strip()

    # ------------------------------------------------------------
    # 1. Extraer email si existe
    # ------------------------------------------------------------
    match_email = re.search(
        r"[\w\.-]+@[\w\.-]+\.\w+",
        texto_original,
        flags=re.IGNORECASE,
    )

    if match_email:
        datos["email"] = match_email.group(0).strip().lower()

    # ------------------------------------------------------------
    # 2. Construir candidato de nombre quitando email y frases comunes
    # ------------------------------------------------------------
    texto_nombre = texto_original

    if match_email:
        texto_nombre = texto_nombre.replace(match_email.group(0), " ")

    patrones_limpieza = [
        r"\bmi\s+nombre\s+es\b",
        r"\bnombre\s+es\b",
        r"\bme\s+llamo\b",
        r"\bsoy\b",
        r"\bmi\s+correo\s+es\b",
        r"\bcorreo\s+es\b",
        r"\bcorreo\s+electr[oó]nico\s+es\b",
        r"\bcorreo\s+electr[oó]nico\b",
        r"\bcorreo\b",
        r"\bemail\s+es\b",
        r"\bemail\b",
        r"\be-mail\s+es\b",
        r"\be-mail\b",
    ]

    for patron in patrones_limpieza:
        texto_nombre = re.sub(
            patron,
            " ",
            texto_nombre,
            flags=re.IGNORECASE,
        )

    # Quitar conectores sueltos que suelen quedar después de remover el correo.
    texto_nombre = re.sub(
        r"\b(y|con|para|al|a)\b",
        " ",
        texto_nombre,
        flags=re.IGNORECASE,
    )

    # Limpiar puntuación y espacios.
    texto_nombre = texto_nombre.strip(" ,.;:-")
    texto_nombre = re.sub(r"\s+", " ", texto_nombre).strip()

    # ------------------------------------------------------------
    # 3. Validar si lo restante parece nombre
    # ------------------------------------------------------------
    nombre = _parece_nombre_simple(texto_nombre)

    if nombre:
        datos["nombre"] = nombre

    return datos

def _extraer_rut_desde_mensaje(mensaje: str) -> Optional[str]:
    """
    Detecta si el cliente está compartiendo el RUT.

    El RUT es documento/soporte tributario, no debe bloquear el flujo.
    Si viene número, lo guardamos. Si solo dice que lo comparte, guardamos 'recibido'.
    """
    texto = (mensaje or "").strip()
    if not texto:
        return None

    texto_norm = _normalizar_intencion(texto)

    if "rut" not in texto_norm:
        return None

    nit_match = NIT_RE.search(texto)
    if nit_match:
        return nit_match.group(1).strip()

    if any(
        frase in texto_norm
        for frase in {
            "rut",
            "te comparto el rut",
            "envio el rut",
            "envie el rut",
            "adjunto el rut",
            "rut adjunto",
            "ya comparti el rut",
            "ya envie el rut",
        }
    ):
        return "recibido"

    return None

def _capturar_dato_comercial_por_etapa(mensaje: str, cliente: dict, etapa: str) -> dict:
    """
    Completa datos comerciales según el estado actual de la conversación.

    Esta función evita que datos como nombre, empresa o NIT sean tratados
    como búsqueda de producto cuando NIA está cerrando una cotización.

    Regla Cotización IV:
    Si NIA pide nombre y correo en un mismo turno, el backend debe poder
    capturar ambos datos correctamente desde una respuesta natural.
    """
    cliente = dict(cliente or {})
    mensaje = (mensaje or "").strip()

    if etapa in {"cotizacion", "calificacion", "confirmando_cierre"}:
        datos_contacto = _extraer_datos_contacto_desde_mensaje(mensaje)

        if not cliente.get("email") and datos_contacto.get("email"):
            cliente["email"] = datos_contacto["email"]
            logger.debug("Email capturado por parser de contacto: %s", cliente["email"])

        if not cliente.get("nombre") and datos_contacto.get("nombre"):
            cliente["nombre"] = datos_contacto["nombre"]
            logger.debug("Nombre capturado por parser de contacto: %s", cliente["nombre"])

        # Fallback para el caso simple:

        if not cliente.get("nombre"):
            nombre = _parece_nombre_simple(mensaje)
            if nombre:
                cliente["nombre"] = nombre
                logger.debug("Nombre simple capturado por etapa: %s", nombre)

    if etapa in {"proforma", "proforma_lista"}:
        if not cliente.get("empresa"):
            empresa = _parece_empresa_simple(mensaje)
            if empresa:
                cliente["empresa"] = empresa
                logger.debug("Empresa simple capturada por etapa: %s", empresa)

        rut = _extraer_rut_desde_mensaje(mensaje)
        if rut and cliente.get("rut") in {None, "", "pendiente"}:
            cliente["rut"] = rut
            logger.debug("RUT capturado/actualizado por parser de RUT: %s", rut)

    return cliente

def _respuesta_siguiente_dato_comercial(
    cliente: dict,
    etapa_objetivo: str = "cotizacion",
) -> tuple[str, str]:
    """
    Decide qué información comercial debe solicitar NIA.

    Reglas v3.66:
    - Cotización: nombre y correo se solicitan en una sola pregunta.
    - Proforma: razón social, NIT y RUT se solicitan en una sola pregunta.
    - No se afirma que el documento fue enviado automáticamente.
    - Después de reunir los datos, NIA espera confirmación externa.
    """
    cliente = dict(cliente or {})

    nombre = str(cliente.get("nombre") or "").strip()
    email = str(cliente.get("email") or "").strip()
    empresa = str(cliente.get("empresa") or "").strip()
    nit = str(cliente.get("nit") or "").strip()
    rut = str(cliente.get("rut") or "").strip()

    # ============================================================
    # ETAPA COTIZACIÓN
    # ============================================================
    if etapa_objetivo in {
        "cotizacion",
        "calificacion",
        "confirmando_cierre",
    }:
        falta_nombre = not nombre
        falta_email = not email

        # Don Andrés: nombre y correo en una sola pregunta.
        if falta_nombre and falta_email:
            return (
                "Para preparar la cotización, indícame por favor "
                "tu nombre y el correo electrónico donde deseas recibirla.",
                "cotizacion",
            )

        if falta_nombre:
            return (
                "Ya tengo el correo. ¿A nombre de quién debe quedar la cotización?",
                "cotizacion",
            )

        if falta_email:
            return (
                f"Gracias, {nombre}. ¿Cuál es el correo electrónico "
                "donde deseas recibir la cotización?",
                "cotizacion",
            )

        # No afirmamos que la cotización ya fue creada o enviada.
        return (
            f"Perfecto, {nombre}. Ya tengo los datos de contacto "
            "para gestionar la cotización. Cuando la recibas, "
            "confírmame si cumple con lo que necesitas técnicamente.",
            "cotizacion_lista",
        )

    # ============================================================
    # ETAPA PROFORMA
    # ============================================================
    if etapa_objetivo == "proforma":
        falta_empresa = not empresa
        falta_nit = not nit

        # "pendiente" no equivale a un RUT recibido.
        falta_rut = not rut or rut == "pendiente"

        # Don Andrés: pedir los datos en una sola pregunta.
        if falta_empresa and falta_nit and falta_rut:
            return (
                "Para preparar la proforma, envíame en un solo mensaje "
                "la razón social, el NIT y el RUT de la empresa. "
                "Puedes adjuntar el RUT o indicar que ya lo compartiste.",
                "proforma",
            )

        faltantes = []

        if falta_empresa:
            faltantes.append("la razón social")

        if falta_nit:
            faltantes.append("el NIT")

        if falta_rut:
            faltantes.append("el RUT")

        if faltantes:
            if len(faltantes) == 1:
                texto_faltantes = faltantes[0]
            elif len(faltantes) == 2:
                texto_faltantes = (
                    f"{faltantes[0]} y {faltantes[1]}"
                )
            else:
                texto_faltantes = (
                    f"{', '.join(faltantes[:-1])} "
                    f"y {faltantes[-1]}"
                )

            return (
                f"Para completar la solicitud de proforma todavía necesito "
                f"{texto_faltantes}. Puedes enviarlo en un solo mensaje.",
                "proforma",
            )

        # No afirmamos que Bitrix o un asesor ya la envió.
        return (
            f"Perfecto, {nombre or 'cliente'}. Ya tengo los datos "
            "necesarios para gestionar la proforma. Cuando la recibas, "
            "confírmame si deseas proceder con el pago.",
            "proforma_lista",
        )

    # Fallback seguro.
    return (
        "Ya tengo la información comercial disponible. "
        "Continuaré únicamente cuando exista una confirmación "
        "real del siguiente documento o etapa.",
        "cotizacion_lista",
    )


def _es_nueva_solicitud_durante_cierre(mensaje: str) -> bool:
    """
    Permite salir del flujo de cierre si el cliente realmente pide otro producto.

    Ejemplo:
    - también necesito una válvula
    - agrega otro sensor
    - necesito otro equipo
    """
    t = _normalizar_intencion(mensaje)

    if any(p in t for p in {"tambien necesito", "tambien quiero", "agrega", "agregar", "otro producto", "otra referencia"}):
        return True

    return _parece_solicitud_de_producto(mensaje)

# ─────────────────────────────────────────────────────────────
# Controlador determinístico de estado comercial
# ─────────────────────────────────────────────────────────────

ESTADOS_COMERCIALES = {
    "producto_encontrado",
    "esperando_cantidad",
    "confirmando_cierre",
    "cotizacion",
    "calificacion",
    "cotizacion_lista",
    "cotizacion_enviada",
    "proforma",
    "proforma_lista",
    "proforma_enviada",
    "pago",
}

def _ultimo_turno_pide_datos_contacto(historial: list) -> bool:
    """
    Detecta si el último mensaje de NIA pidió datos básicos de contacto.

    Regla de negocio:
    Si NIA acaba de pedir nombre/correo, el siguiente mensaje del cliente
    debe ser tratado como dato comercial de cotización, no como búsqueda
    de catálogo.

    Esto protege el flujo cuando la etapa persistida queda inconsistente.
    """
    if not historial:
        return False

    # Buscar el último mensaje del asistente.
    ultimo_assistant = None

    for turno in reversed(historial):
        if turno.get("role") == "assistant":
            ultimo_assistant = turno.get("content", "")
            break

    if not ultimo_assistant:
        return False

    texto = _normalizar_intencion(ultimo_assistant)

    indicadores_contacto = [
        "nombre y correo",
        "nombre y correo electronico",
        "nombre y e mail",
        "a nombre de quien",
        "correo electronico",
        "correo para enviar la cotizacion",
        "dejar la solicitud lista",
        "datos basicos",
    ]

    return any(indicador in texto for indicador in indicadores_contacto)

def _manejar_estado_comercial_prioritario(
    etapa: str,
    mensaje: str,
    cliente: dict,
    productos_acumulados: list,
    necesidad_ctx: dict,
    clasificacion: Optional[dict] = None,
) -> Optional[dict]:
    """
    Controla estados comerciales de forma determinística.

    Regla central:
    Si esta función resuelve el turno, procesar_turno debe retornar
    inmediatamente sin buscar catálogo y sin llamar al LLM.

    Esto evita que:
    - una cantidad sea interpretada como código;
    - un NIT sea interpretado como producto;
    - un correo dispare búsqueda de catálogo;
    - el LLM cambie una etapa comercial ya decidida.
    """
    etapa = etapa or "inicio"
    mensaje = (mensaje or "").strip()
    cliente = dict(cliente or {})
    necesidad_ctx = dict(necesidad_ctx or {})
    productos_acumulados = productos_acumulados or []

    clasificacion = clasificacion or {}
    tipo_mensaje = clasificacion.get("tipo")

    if not mensaje or etapa not in ESTADOS_COMERCIALES:
        return None

    # ============================================================
    # Cotización recibida por el cliente
    # ============================================================
    if tipo_mensaje in {"cotizacion_recibida", "link_documento"} and etapa in {
        "cotizacion_lista",
        "cotizacion",
        "confirmando_cierre",
    }:
        necesidad_ctx["cotizacion_recibida"] = True

        if tipo_mensaje == "link_documento":
            necesidad_ctx["archivo_cotizacion"] = mensaje

        return {
            "handled": True,
            "respuesta": (
                "Perfecto, tomo esto como cotización recibida. "
                "¿La cotización cumple con lo que necesitas técnicamente?"
            ),
            "etapa": "cotizacion_enviada",
            "cliente": cliente,
            "necesidad_ctx": necesidad_ctx,
            "productos_acumulados": productos_acumulados,
        }

    # ============================================================
    # Cliente valida cotización enviada
    # ============================================================
    if etapa == "cotizacion_enviada":
        if _es_confirmacion_afirmativa(mensaje):
            necesidad_ctx["cotizacion_aprobada_cliente"] = True

            respuesta_dato, etapa_dato = _respuesta_siguiente_dato_comercial(
                cliente,
                etapa_objetivo="proforma",
            )

            return {
                "handled": True,
                "respuesta": respuesta_dato,
                "etapa": etapa_dato,
                "cliente": cliente,
                "necesidad_ctx": necesidad_ctx,
                "productos_acumulados": productos_acumulados,
            }

        if _es_confirmacion_negativa(mensaje):
            necesidad_ctx["cotizacion_aprobada_cliente"] = False

            return {
                "handled": True,
                "respuesta": (
                    "Entiendo. Cuéntame qué ajuste necesitas en la cotización "
                    "o qué característica técnica no cumple."
                ),
                "etapa": "descubrimiento",
                "cliente": cliente,
                "necesidad_ctx": necesidad_ctx,
                "productos_acumulados": productos_acumulados,
            }

        return {
            "handled": True,
            "respuesta": (
                "Para avanzar correctamente, necesito confirmar: "
                "¿la cotización cumple con lo que necesitas técnicamente?"
            ),
            "etapa": "cotizacion_enviada",
            "cliente": cliente,
            "necesidad_ctx": necesidad_ctx,
            "productos_acumulados": productos_acumulados,
        }

    # ============================================================
    # Proforma recibida por el cliente
    # ============================================================
    if tipo_mensaje in {"proforma_recibida", "link_documento"} and etapa in {
        "proforma",
        "proforma_lista",
        "cotizacion_enviada",
    }:
        necesidad_ctx["proforma_recibida"] = True

        if tipo_mensaje == "link_documento":
            necesidad_ctx["archivo_proforma"] = mensaje

        return {
            "handled": True,
            "respuesta": (
                "Perfecto, tomo esto como proforma recibida. "
                "¿Deseas proceder con el pago?"
            ),
            "etapa": "proforma_enviada",
            "cliente": cliente,
            "necesidad_ctx": necesidad_ctx,
            "productos_acumulados": productos_acumulados,
        }

    # ============================================================
    # Cliente valida proforma para pago
    # ============================================================
    if etapa == "proforma_enviada":
        if _es_confirmacion_afirmativa(mensaje):
            necesidad_ctx["proforma_aprobada_cliente"] = True
            return {
                "handled": True,
                "respuesta": (
                    "Perfecto. ¿Qué medio de pago prefieres: "
                    "PSE, transferencia o tarjeta?"
                ),
                "etapa": "pago",
                "cliente": cliente,
                "necesidad_ctx": necesidad_ctx,
                "productos_acumulados": productos_acumulados,
            }

        if _es_confirmacion_negativa(mensaje):
            return {
                "handled": True,
                "respuesta": (
                    "Entiendo. Indícame qué ajuste necesitas en la proforma "
                    "para que el asesor pueda revisarlo."
                ),
                "etapa": "proforma",
                "cliente": cliente,
                "necesidad_ctx": necesidad_ctx,
                "productos_acumulados": productos_acumulados,
            }

        return {
            "handled": True,
            "respuesta": "¿Deseas proceder con el pago?",
            "etapa": "proforma_enviada",
            "cliente": cliente,
            "necesidad_ctx": necesidad_ctx,
            "productos_acumulados": productos_acumulados,
        }

    # ============================================================
    # Selección del medio de pago
    # ============================================================
    if etapa == "pago":
        texto_pago = _normalizar_intencion(mensaje)

        medio_pago = None

        if "pse" in texto_pago:
            medio_pago = "PSE"
        elif "transferencia" in texto_pago:
            medio_pago = "transferencia"
        elif "tarjeta" in texto_pago:
            medio_pago = "tarjeta"

        if not medio_pago:
            return {
                "handled": True,
                "respuesta": (
                    "Para continuar, indícame cuál medio de pago prefieres: "
                    "PSE, transferencia o tarjeta."
                ),
                "etapa": "pago",
                "cliente": cliente,
                "necesidad_ctx": necesidad_ctx,
                "productos_acumulados": productos_acumulados,
            }

        necesidad_ctx["medio_pago"] = medio_pago
        necesidad_ctx["medio_pago_confirmado"] = True

        return {
            "handled": True,
            "respuesta": (
                f"Perfecto. Registré {medio_pago} como medio de pago preferido. "
                "Continúa únicamente mediante el canal oficial indicado "
                "en la proforma."
            ),
            "etapa": "pago",
            "cliente": cliente,
            "necesidad_ctx": necesidad_ctx,
            "productos_acumulados": productos_acumulados,
        }

    # ============================================================
    # Proforma lista, esperando confirmación externa
    # ============================================================
    if etapa == "proforma_lista":
        cliente = _capturar_dato_comercial_por_etapa(
            mensaje=mensaje,
            cliente=cliente,
            etapa="proforma_lista",
        )

        rut_valor = _extraer_rut_desde_mensaje(mensaje)
        if rut_valor and cliente.get("rut") in {None, "", "pendiente"}:
            cliente["rut"] = rut_valor
            logger.debug("RUT actualizado en proforma_lista: %s", rut_valor)

        if rut_valor:
            return {
                "handled": True,
                "respuesta": (
                        "Gracias, recibí el RUT y lo dejé asociado a tu solicitud. "
                        "En breve recibirás la proforma."
                ),
                "etapa": "proforma_lista",
                "cliente": cliente,
                "necesidad_ctx": necesidad_ctx,
                "productos_acumulados": productos_acumulados,
            }

        return {
            "handled": True,
            "respuesta": (
                "Tu proforma está en proceso. Cuando la recibas, me confirmas si deseas proceder con el pago."
            ),
            "etapa": "proforma_lista",
            "cliente": cliente,
            "necesidad_ctx": necesidad_ctx,
            "productos_acumulados": productos_acumulados,
        }



    # 1) Producto encontrado: NIA espera confirmación explícita.
    if etapa == "producto_encontrado":
        if _es_confirmacion_afirmativa(mensaje):
            return {
                "handled": True,
                "respuesta": "Perfecto. ¿Cuál es la cantidad que necesitas?",
                "etapa": "esperando_cantidad",
                "cliente": cliente,
                "necesidad_ctx": {"esperando": "cantidad"},
                "productos_acumulados": productos_acumulados,
            }

        if _es_confirmacion_negativa(mensaje):
            return {
                "handled": True,
                "respuesta": (
                    "Entendido. Para buscar una mejor opción, ¿puedes indicarme "
                    "tipo de producto, aplicación, marca, referencia o especificación técnica requerida?"
                ),
                "etapa": "descubrimiento",
                "cliente": cliente,
                "necesidad_ctx": {},
                "productos_acumulados": productos_acumulados,
            }

        return {
            "handled": True,
            "respuesta": "¿Este producto cubre lo que necesitas? Puedes responder sí o no.",
            "etapa": "producto_encontrado",
            "cliente": cliente,
            "necesidad_ctx": necesidad_ctx,
            "productos_acumulados": productos_acumulados,
        }

    # 2) Esperando cantidad: un número corto es cantidad, no código.
    if etapa == "esperando_cantidad":
        cantidad = _extraer_cantidad_solicitada(mensaje)

        if not cantidad:
            return {
                "handled": True,
                "respuesta": "Para avanzar con la cotización necesito la cantidad en unidades. ¿Cuántas unidades necesitas?",
                "etapa": "esperando_cantidad",
                "cliente": cliente,
                "necesidad_ctx": {"esperando": "cantidad"},
                "productos_acumulados": productos_acumulados,
            }

        _asignar_cantidad_ultimo_producto(productos_acumulados, cantidad)

        return {
            "handled": True,
            "respuesta": f"Listo, dejé la cantidad en {cantidad}. ¿Necesitas algo más o cotizamos con esto?",
            "etapa": "confirmando_cierre",
            "cliente": cliente,
            "necesidad_ctx": {},
            "productos_acumulados": productos_acumulados,
        }

    # 3) Confirmando cierre: por defecto sigue a datos comerciales.
    # Solo sale a catálogo si el cliente claramente pide otro producto.
    if etapa == "confirmando_cierre":
        if _es_nueva_solicitud_durante_cierre(mensaje):
            return None

        cliente = _capturar_dato_comercial_por_etapa(
            mensaje=mensaje,
            cliente=cliente,
            etapa=etapa,
        )

        respuesta_dato, etapa_dato = _respuesta_siguiente_dato_comercial(
            cliente,
            etapa_objetivo="cotizacion",
        )

        return {
            "handled": True,
            "respuesta": respuesta_dato,
            "etapa": etapa_dato,
            "cliente": cliente,
            "necesidad_ctx": {},
            "productos_acumulados": productos_acumulados,
        }

    # 4) Cotización/proforma: capturar datos antes de cualquier búsqueda.
    if etapa in {"cotizacion", "calificacion", "proforma"}:
        if _es_nueva_solicitud_durante_cierre(mensaje):
            return None

        cliente = _capturar_dato_comercial_por_etapa(
            mensaje=mensaje,
            cliente=cliente,
            etapa=etapa,
        )

        etapa_objetivo = "proforma" if etapa == "proforma" else "cotizacion"

        respuesta_dato, etapa_dato = _respuesta_siguiente_dato_comercial(
            cliente,
            etapa_objetivo=etapa_objetivo,
        )

        return {
            "handled": True,
            "respuesta": respuesta_dato,
            "etapa": etapa_dato,
            "cliente": cliente,
            "necesidad_ctx": {},
            "productos_acumulados": productos_acumulados,
        }

    # 5) Cierre seguro: no inventar cotización ni proforma automática.
    if etapa == "cotizacion_lista":
        return {
            "handled": True,
            "respuesta": (
                "Tu cotización está en proceso. Cuando la recibas, me confirmas si cumple con lo que necesitas técnicamente."
            ),
            "etapa": "cotizacion_lista",
            "cliente": cliente,
            "necesidad_ctx": {},
            "productos_acumulados": productos_acumulados,
        }

    return None

async def _persistir_cliente_permanente(
    phone_id: Optional[str],
    cliente: Optional[dict],
) -> None:
    """
    Guarda datos comerciales del cliente en memoria permanente.

    Esta memoria NO reemplaza la sesión conversacional.
    Solo persiste datos reutilizables del cliente:
    - nombre
    - email
    - empresa
    - nit
    - rut
    - teléfono/phone_id

    Si Mongo falla, no rompemos la conversación del cliente.
    """
    if not phone_id or not cliente:
        return

    try:
        await upsert_cliente(phone_id, cliente)
    except Exception as e:
        logger.warning(
            "No fue posible persistir cliente permanente phone_id=%s error=%s",
            phone_id,
            e,
        )

async def _guardar_y_responder_turno(
    session_id: str,
    phone_id: Optional[str],
    historial: list,
    mensaje_usuario: str,
    respuesta: str,
    etapa: str,
    cliente: dict,
    productos_acumulados: list,
    necesidad_ctx: Optional[dict] = None,
    archivo_activo: Optional[dict] = None,
    items_resultado: Optional[list] = None,
    cotizacion_recibida: bool = False,
    archivo_cotizacion: Optional[str] = None,
    proforma_recibida: bool = False,
    archivo_proforma: Optional[str] = None,
):
    """
    Guarda sesión y retorna respuesta final sin pasar por LLM.

    Se usa cuando un estado comercial ya resolvió el turno.
    """
    turno_user = {
        "role": "user",
        "content": mensaje_usuario,
        "ts": datetime.utcnow().isoformat(),
    }

    turno_nia = {
        "role": "assistant",
        "content": respuesta,
        "ts": datetime.utcnow().isoformat(),
    }

    await _persistir_cliente_permanente(phone_id, cliente)

    await save_session(
        session_id=session_id,
        phone_id=phone_id,
        turnos=historial + [turno_user, turno_nia],
        etapa=etapa,
        archivo_activo=archivo_activo,
        necesidad_ctx=necesidad_ctx or {},
        cliente=cliente or {},
        productos_acumulados=productos_acumulados or [],
        cotizacion_recibida=cotizacion_recibida,
        archivo_cotizacion=archivo_cotizacion,
        proforma_recibida=proforma_recibida,
        archivo_proforma=archivo_proforma,
    )

    return {
        "respuesta": respuesta,
        "etapa": etapa,
        "items_resultado": items_resultado or None,
        "cliente": cliente or None,
    }

# ─────────────────────────────────────────────────────────────
# Núcleo conversacional
# ─────────────────────────────────────────────────────────────

async def procesar_turno(
    session_id: str,
    mensaje: str,
    phone_id: Optional[str] = None,
    archivo_bytes: Optional[bytes] = None,
    archivo_nombre: Optional[str] = None,
) -> dict:
    logger.info(
        "Turno: session=%s etapa_msg='%s'",
        session_id,
        mensaje[:50] if mensaje else "[archivo]",
    )

    session = await get_session(session_id) or {}

    historial = session.get("turnos", [])
    etapa = session.get("etapa", "inicio")
        # ------------------------------------------------------------
    # Guardrail de estado comercial:
    # Si el último mensaje de NIA pidió datos de contacto, el
    # siguiente mensaje del cliente debe procesarse como cotización,
    # aunque la etapa guardada haya quedado inconsistente.
    # ------------------------------------------------------------
    if etapa in {"inicio", "descubrimiento"} and _ultimo_turno_pide_datos_contacto(historial):
            logger.info(
                "Corrigiendo etapa por último turno de contacto: session=%s etapa=%s -> cotizacion",
                session_id,
                etapa,
            )
            etapa = "cotizacion"
    archivo_activo = session.get("archivo_activo")
    necesidad_ctx = session.get("necesidad_ctx", {})

    # ------------------------------------------------------------
    # Cliente: sesión temporal + memoria permanente
    # ------------------------------------------------------------
    # La sesión tiene prioridad porque contiene lo más reciente
    # dentro de la conversación actual.
    cliente_sesion = session.get("cliente", {}) or {}

    cliente_permanente = {}

    if phone_id:
        try:
            cliente_permanente = await get_cliente(phone_id) or {}
        except Exception as e:
            logger.warning(
                "No fue posible cargar cliente permanente phone_id=%s error=%s",
                phone_id,
                e,
            )
            cliente_permanente = {}

    cliente = {
        **cliente_permanente,
        **cliente_sesion,
    }

    productos_acumulados = session.get("productos_acumulados", [])

    cotizacion_recibida = bool(session.get("cotizacion_recibida", False))
    archivo_cotizacion = session.get("archivo_cotizacion")
    proforma_recibida = bool(session.get("proforma_recibida", False))
    archivo_proforma = session.get("archivo_proforma")

    contexto_extra = ""
    nueva_etapa = etapa
    items_resultado = []

    if mensaje.strip():
        cliente = extraer_datos_cliente(mensaje, cliente)

    # ------------------------------------------------------------
    # Clasificación de intención — Cotización V
    # ------------------------------------------------------------
    clasificacion = await clasificar_mensaje(mensaje, etapa)

    logger.info(
        "Clasificación mensaje: session=%s etapa=%s tipo=%s confianza=%s razon=%s",
        session_id,
        etapa,
        clasificacion.get("tipo"),
        clasificacion.get("confianza"),
        clasificacion.get("razon"),
    )

    # ══════════════════════════════════════════════════════
    # PRIORIDAD ABSOLUTA: ESTADO COMERCIAL
    # ══════════════════════════════════════════════════════
    # Si el turno pertenece a una etapa comercial, se resuelve aquí
    # y se retorna inmediatamente. No catálogo. No LLM. No reglas posteriores.
    if mensaje.strip() and not (archivo_bytes and archivo_nombre):
        comercial = _manejar_estado_comercial_prioritario(
            etapa=etapa,
            mensaje=mensaje,
            cliente=cliente,
            productos_acumulados=productos_acumulados,
            necesidad_ctx=necesidad_ctx,
            clasificacion=clasificacion,
        )

        if comercial and comercial.get("handled"):
            logger.info(
                "Turno resuelto por estado comercial: session=%s etapa=%s -> %s",
                session_id,
                etapa,
                comercial["etapa"],
            )

            necesidad_comercial = comercial.get("necesidad_ctx", {}) or {}

            return await _guardar_y_responder_turno(
                session_id=session_id,
                phone_id=phone_id,
                historial=historial,
                mensaje_usuario=mensaje,
                respuesta=comercial["respuesta"],
                etapa=comercial["etapa"],
                cliente=comercial["cliente"],
                productos_acumulados=comercial["productos_acumulados"],
                necesidad_ctx=necesidad_comercial,
                archivo_activo=archivo_activo,
                items_resultado=None,
                cotizacion_recibida=bool(
                    necesidad_comercial.get("cotizacion_recibida", cotizacion_recibida)
                ),
                archivo_cotizacion=necesidad_comercial.get(
                    "archivo_cotizacion", archivo_cotizacion
                ),
                proforma_recibida=bool(
                    necesidad_comercial.get("proforma_recibida", proforma_recibida)
                ),
                archivo_proforma=necesidad_comercial.get(
                    "archivo_proforma", archivo_proforma
                ),
            )

    # ══════════════════════════════════════════════════════
    # MODO ARCHIVO
    # ══════════════════════════════════════════════════════

    if archivo_bytes and archivo_nombre:
        logger.info("Procesando archivo: %s", archivo_nombre)

        items = await procesar_archivo(archivo_bytes, archivo_nombre)

        for item in items:
            tipo, valor = detectar_identificador(item["texto"])

            if tipo:
                res = await rama_codigo(valor, tipo)
            else:
                nec = await evaluar_necesidad(item["texto"])

                if nec["clara"]:
                    res = await buscar_en_catalogo(item["texto"])

                    if debe_intentar_enriquecimiento(res):
                        res = await enriquecer_y_buscar(item["texto"])

                else:
                    res = {
                        "estado": "pendiente",
                        "preguntas": nec["preguntas"],
                    }

            res["texto_original"] = item["texto"]
            res["fila"] = item.get("fila")
            res["cantidad"] = item.get("cantidad")
            items_resultado.append(res)

            if res["estado"] == "encontrado" and res.get("producto"):
                productos_acumulados.append({
                    "producto": res["producto"],
                    "cantidad": item.get("cantidad"),
                    "desde": "archivo",
                    "ts": datetime.utcnow().isoformat(),
                })

        encontrados = [i for i in items_resultado if i["estado"] == "encontrado"]
        pendientes = [
            i for i in items_resultado
            if i["estado"] in {"pendiente", "sin_resultado", "relacionado"}
        ]

        archivo_activo = {
            "nombre": archivo_nombre,
            "total_items": len(items),
            "items": items_resultado,
            "ts": datetime.utcnow().isoformat(),
        }

        nueva_etapa = "procesando_archivo"

        resumen = (
            f"[ARCHIVO: {archivo_nombre}]\n"
            f"Total: {len(items)} · Encontrados: {len(encontrados)} · "
            f"Pendientes/por validar: {len(pendientes)}\n"
        )

        for item_resultado in items_resultado:
            resumen += f"- {item_resultado['texto_original']}: {item_resultado['estado'].upper()}"

            if item_resultado.get("producto"):
                p = item_resultado["producto"]
                resumen += f" → {p.get('codigo')} | {p.get('nombre')}"

                if item_resultado["estado"] == "relacionado":
                    resumen += " [RELACIONADO — REQUIERE CONFIRMACIÓN]"
                elif not item_resultado.get("exacto", True):
                    resumen += " [COINCIDENCIA CERCANA]"

            resumen += "\n"

        contexto_extra = resumen

    # ══════════════════════════════════════════════════════
    # MODO TEXTO
    # ══════════════════════════════════════════════════════

    elif mensaje.strip():
        msg_lower = mensaje.lower().strip()
        estado_comercial_resuelto = False

        # El estado comercial se resuelve antes de entrar al modo texto.
        # Si llegó hasta aquí, este turno puede pasar a archivo/catálogo/LLM.

        # Caso 1: respuesta a ítem pendiente de archivo
        if archivo_activo:
            pendientes = [
                item for item in archivo_activo.get("items", [])
                if item["estado"] in {"pendiente", "sin_resultado", "relacionado"}
            ]

            if pendientes:
                item_pend = pendientes[0]
                query_e = f"{item_pend['texto_original']} {mensaje}".strip()

                res = await buscar_en_catalogo(query_e)

                if debe_intentar_enriquecimiento(res):
                    res = await enriquecer_y_buscar(query_e)

                res["texto_original"] = item_pend["texto_original"]

                for idx, item in enumerate(archivo_activo["items"]):
                    if item["texto_original"] == item_pend["texto_original"]:
                        archivo_activo["items"][idx] = res
                        break

                contexto_extra, nueva_etapa, necesidad_ctx = construir_respuesta_desde_resultado(
                    res=res,
                    cliente=cliente,
                    productos_acumulados=productos_acumulados,
                    desde="archivo_pendiente",
                    necesidad_ctx_base={
                        "texto_original": item_pend["texto_original"],
                        "query_evaluada": query_e,
                    },
                )

        # Caso 2: validación de producto relacionado
        elif etapa == "validando_relacionado" and necesidad_ctx.get("producto_relacionado"):
            producto_relacionado = necesidad_ctx["producto_relacionado"]

            if any(palabra in msg_lower for palabra in {"sí", "si", "correcto", "ese", "me sirve", "sirve"}):
                productos_acumulados.append({
                    "producto": producto_relacionado,
                    "cantidad": None,
                    "desde": "confirmacion_relacionado",
                    "ts": datetime.utcnow().isoformat(),
                })

                contexto_extra = _marcar_respuesta_segura(
                    respuesta_producto_encontrado(producto_relacionado, cliente)
                )
                nueva_etapa = "producto_encontrado"
                necesidad_ctx = {}

            else:
                texto_original = (
                    necesidad_ctx.get("texto_original")
                    or necesidad_ctx.get("query_evaluada")
                    or mensaje
                )
                query_e = f"{texto_original} {mensaje}".strip()

                res = await buscar_en_catalogo(query_e)

                if debe_intentar_enriquecimiento(res):
                    res = await enriquecer_y_buscar(query_e)

                contexto_extra, nueva_etapa, necesidad_ctx = construir_respuesta_desde_resultado(
                    res=res,
                    cliente=cliente,
                    productos_acumulados=productos_acumulados,
                    desde="validacion_relacionado",
                    necesidad_ctx_base={
                        "texto_original": texto_original,
                        "query_evaluada": query_e,
                    },
                )

                # Caso 2B: respuesta a pregunta por tipo de producto
        # ------------------------------------------------------------
        # Flujo Grupo C:
        #
        # 1. El cliente ya indicó la familia genérica.
        # 2. NIA mostró tipos reales del catálogo.
        # 3. El cliente acaba de confirmar un tipo.
        # 4. Antes de presentar un SKU, analizamos los campos técnicos
        #    reales de los candidatos coherentes.
        # 5. Se seleccionan máximo dos campos discriminantes.
        # 6. Se pregunta uno por turno.
        #
        # Máximo total:
        # - pregunta de tipo;
        # - campo técnico 1;
        # - campo técnico 2.
        # ------------------------------------------------------------
        elif (
            etapa == "esperando_tipo_producto"
            and necesidad_ctx.get("tipos_detectados")
        ):
            tipos_detectados_previos = necesidad_ctx.get(
                "tipos_detectados",
                [],
            )

            texto_original = (
                necesidad_ctx.get("texto_original")
                or necesidad_ctx.get("query_evaluada")
                or ""
            )

            tipo_confirmado = mensaje.strip()

            query_tipo = " ".join(
                parte
                for parte in [
                    texto_original,
                    tipo_confirmado,
                ]
                if str(parte or "").strip()
            ).strip()

            productos_candidatos = []
            productos_coherentes = []
            campos_disponibles = []
            campos_seleccionados = []
            preguntas_dinamicas = []

            try:
                # ----------------------------------------------------
                # 1. Recuperar candidatos del tipo elegido
                # ----------------------------------------------------
                productos_candidatos = (
                    await buscar_por_texto(query_tipo)
                    or []
                )

                # ----------------------------------------------------
                # 2. Eliminar productos de familias no coherentes
                # ----------------------------------------------------
                productos_coherentes = filtrar_candidatos_coherentes(
                    texto_cliente=query_tipo,
                    productos=productos_candidatos,
                )

                # Fallback defensivo:
                # si el filtro fue demasiado restrictivo, conservamos
                # los candidatos reales de la búsqueda, pero nunca
                # inventamos productos.
                if not productos_coherentes:
                    productos_coherentes = productos_candidatos

                # ----------------------------------------------------
                # 3. Analizar campos técnicos reales
                # ----------------------------------------------------
                campos_disponibles = campos_disponibles_de(
                    productos=productos_coherentes,
                    min_cobertura=0.10,
                    min_valores_distintos=2,
                    max_campos=10,
                )

                # ----------------------------------------------------
                # 4. Seleccionar máximo dos campos discriminantes
                # ----------------------------------------------------
                campos_seleccionados = ordenar_campos_por_prioridad(
                    campos_disponibles=campos_disponibles,
                    max_campos=2,
                )

                # ----------------------------------------------------
                # 5. Convertir los campos en preguntas naturales
                # ----------------------------------------------------
                preguntas_dinamicas = (
                    generar_preguntas_campos_dinamicos(
                        texto_producto=query_tipo,
                        campos_seleccionados=campos_seleccionados,
                        max_preguntas=2,
                    )
                )

                preguntas_dinamicas = _limpiar_preguntas_tecnicas(
                    preguntas_dinamicas
                )

                logger.info(
                    "Flujo campos dinámicos: query='%s' "
                    "candidatos=%s coherentes=%s campos=%s preguntas=%s",
                    query_tipo,
                    len(productos_candidatos),
                    len(productos_coherentes),
                    [
                        campo.get("campo")
                        for campo in campos_seleccionados
                        if isinstance(campo, dict)
                    ],
                    len(preguntas_dinamicas),
                )

            except Exception as e:
                logger.exception(
                    "Error construyendo campos dinámicos "
                    "después de confirmar tipo: %s",
                    e,
                )

                productos_candidatos = []
                productos_coherentes = []
                campos_disponibles = []
                campos_seleccionados = []
                preguntas_dinamicas = []

            # --------------------------------------------------------
            # Si existen campos reales, iniciamos preguntas secuenciales
            # --------------------------------------------------------
            if preguntas_dinamicas:
                # Guardamos solo un resumen liviano de los campos.
                # No guardamos los productos completos en la sesión.
                campos_ctx = []

                for campo in campos_seleccionados:
                    if not isinstance(campo, dict):
                        continue

                    campos_ctx.append(
                        {
                            "campo": campo.get("campo"),
                            "familia_semantica": campo.get(
                                "familia_semantica"
                            ),
                            "score": campo.get("score"),
                            "cobertura": campo.get("cobertura"),
                        }
                    )

                necesidad_ctx = _crear_ctx_preguntas_secuenciales(
                    texto_original=query_tipo,
                    preguntas=preguntas_dinamicas,
                    necesidad_ctx_base={
                        "texto_original": query_tipo,
                        "query_evaluada": query_tipo,
                        "texto_producto_base": texto_original,
                        "tipos_detectados": tipos_detectados_previos,
                        "tipo_confirmado": tipo_confirmado,

                        # Identifica que estas preguntas provienen
                        # de campos reales del catálogo.
                        "flujo_campos_dinamicos": True,
                        "campos_tecnicos_seleccionados": campos_ctx,

                        # Métricas útiles para logs y auditoría.
                        "cantidad_candidatos": len(
                            productos_candidatos
                        ),
                        "cantidad_coherentes": len(
                            productos_coherentes
                        ),
                    },
                )

                contexto_extra = _marcar_respuesta_segura(
                    _respuesta_pregunta_tecnica_unica(
                        preguntas_dinamicas[0],
                        primera=True,
                    )
                )

                nueva_etapa = "descubrimiento"

            # --------------------------------------------------------
            # Si el catálogo no tiene campos estructurados suficientes,
            # no inventamos preguntas: buscamos directamente.
            # --------------------------------------------------------
            else:
                res = await buscar_en_catalogo(query_tipo)

                if debe_intentar_enriquecimiento(res):
                    res = await enriquecer_y_buscar(query_tipo)

                contexto_extra, nueva_etapa, necesidad_ctx = (
                    construir_respuesta_desde_resultado(
                        res=res,
                        cliente=cliente,
                        productos_acumulados=productos_acumulados,
                        desde="tipo_producto_sin_campos",
                        necesidad_ctx_base={
                            "texto_original": query_tipo,
                            "query_evaluada": query_tipo,
                            "texto_producto_base": texto_original,
                            "tipos_detectados": tipos_detectados_previos,
                            "tipo_confirmado": tipo_confirmado,
                            "flujo_campos_dinamicos": False,
                        },
                    )
                )

        # Caso 3A: preguntas técnicas secuenciales
        # NIA tiene hasta 3 preguntas guardadas, pero solo muestra una por turno.
        elif etapa == "descubrimiento" and necesidad_ctx.get("preguntas_tecnicas"):
            preguntas = _limpiar_preguntas_tecnicas(
                necesidad_ctx.get("preguntas_tecnicas", [])
            )

            texto_original = (
                necesidad_ctx.get("texto_original")
                or necesidad_ctx.get("query_evaluada")
                or ""
            )

            try:
                pregunta_actual_idx = int(necesidad_ctx.get("pregunta_actual_idx", 0))
            except (TypeError, ValueError):
                pregunta_actual_idx = 0

            respuestas_tecnicas = necesidad_ctx.get("respuestas_tecnicas", [])

            if not isinstance(respuestas_tecnicas, list):
                respuestas_tecnicas = []

            respuestas_tecnicas = [
                str(r).strip()
                for r in respuestas_tecnicas
                if str(r).strip()
            ]

            if mensaje.strip():
                respuestas_tecnicas.append(mensaje.strip())

            siguiente_idx = pregunta_actual_idx + 1

            if preguntas and siguiente_idx < len(preguntas):
                contexto_extra = _marcar_respuesta_segura(
                    _respuesta_pregunta_tecnica_unica(
                        preguntas[siguiente_idx],
                        primera=False,
                    )
                )

                nueva_etapa = "descubrimiento"
                necesidad_ctx = {
                    **necesidad_ctx,
                    "texto_original": texto_original,
                    "preguntas_tecnicas": preguntas,
                    "pregunta_actual_idx": siguiente_idx,
                    "respuestas_tecnicas": respuestas_tecnicas,
                }

            else:
                # ----------------------------------------------------
                # Construir la consulta final
                # ----------------------------------------------------
                # Para preguntas dinámicas asociamos cada respuesta
                # al nombre del campo real del catálogo.
                # ----------------------------------------------------
                if necesidad_ctx.get("flujo_campos_dinamicos"):
                    campos_seleccionados_ctx = (
                        necesidad_ctx.get(
                            "campos_tecnicos_seleccionados",
                            [],
                        )
                        or []
                    )

                    partes_query = [texto_original]

                    for indice, respuesta_tecnica in enumerate(
                        respuestas_tecnicas
                    ):
                        campo_nombre = ""

                        if indice < len(campos_seleccionados_ctx):
                            campo_info = (
                                campos_seleccionados_ctx[indice]
                            )

                            if isinstance(campo_info, dict):
                                campo_nombre = str(
                                    campo_info.get("campo") or ""
                                ).strip()

                        if campo_nombre:
                            partes_query.append(
                                f"{campo_nombre}: "
                                f"{respuesta_tecnica}"
                            )
                        else:
                            partes_query.append(
                                respuesta_tecnica
                            )

                    query_e = " ".join(
                        parte
                        for parte in partes_query
                        if str(parte or "").strip()
                    ).strip()

                else:
                    # Flujo tradicional existente.
                    query_e = " ".join(
                        [texto_original] + respuestas_tecnicas
                    ).strip()

                res = await buscar_en_catalogo(query_e)

                if debe_intentar_enriquecimiento(res):
                    res = await enriquecer_y_buscar(query_e)

                contexto_extra, nueva_etapa, necesidad_ctx = construir_respuesta_desde_resultado(
                    res=res,
                    cliente=cliente,
                    productos_acumulados=productos_acumulados,
                    desde="descubrimiento_secuencial",
                    necesidad_ctx_base={
                        "texto_original": texto_original,
                        "query_evaluada": query_e,
                        "diagnostico_tecnico_completo": True,
                        "flujo_campos_dinamicos": bool(
                            necesidad_ctx.get(
                                "flujo_campos_dinamicos",
                                False,
                            )
                        ),
                    },
                )

        # Caso 3: respuesta a preguntas de descubrimiento
        elif etapa == "descubrimiento" and necesidad_ctx.get("texto_original"):
            query_e = f"{necesidad_ctx['texto_original']} {mensaje}".strip()

            nec = await evaluar_necesidad(query_e)

            if nec["clara"]:
                res = await buscar_en_catalogo(query_e)

                if debe_intentar_enriquecimiento(res):
                    res = await enriquecer_y_buscar(query_e)

                contexto_extra, nueva_etapa, necesidad_ctx = construir_respuesta_desde_resultado(
                    res=res,
                    cliente=cliente,
                    productos_acumulados=productos_acumulados,
                    desde="descubrimiento",
                    necesidad_ctx_base={
                        "texto_original": necesidad_ctx.get("texto_original"),
                        "query_evaluada": query_e,
                    },
                )

            else:
                preguntas = _limpiar_preguntas_tecnicas(nec.get("preguntas", []))

                if preguntas:
                    texto_original = necesidad_ctx.get("texto_original")

                    necesidad_ctx = _crear_ctx_preguntas_secuenciales(
                        texto_original=texto_original,
                        preguntas=preguntas,
                        necesidad_ctx_base={
                            "texto_original": texto_original,
                        },
                    )

                    contexto_extra = _marcar_respuesta_segura(
                        _respuesta_pregunta_tecnica_unica(
                            preguntas[0],
                            primera=True,
                        )
                    )
                else:
                    contexto_extra = _marcar_respuesta_segura(
                        "Necesito un poco más de información para buscar la opción adecuada. "
                        "¿Puedes indicarme la aplicación o especificación principal?"
                    )
                    necesidad_ctx = {
                        "texto_original": necesidad_ctx.get("texto_original")
                    }

                nueva_etapa = "descubrimiento"

        # Caso 4: solo saludo
        elif es_solo_saludo(mensaje):
            contexto_extra = (
                f"{saludo_personalizado(cliente)}\n"
                "[SALUDA Y PREGUNTA QUÉ NECESITA O SI TIENE CÓDIGO]"
            )
            nueva_etapa = "saludo"

        # Caso 5: código, referencia o descripción nueva
        else:
            tipo, valor = detectar_identificador(mensaje)

            if tipo:
                res = await rama_codigo(valor, tipo)

                contexto_extra, nueva_etapa, necesidad_ctx = construir_respuesta_desde_resultado(
                    res=res,
                    cliente=cliente,
                    productos_acumulados=productos_acumulados,
                    desde="codigo",
                    necesidad_ctx_base={
                        "texto_original": mensaje,
                        "query_evaluada": valor,
                    },
                )

            else:
                productos_tipo = await buscar_por_texto(mensaje) or []

                debe_preguntar_tipo, tipos_detectados = debe_preguntar_tipo_producto(
                    texto_cliente=mensaje,
                    productos=productos_tipo,
                )

                if debe_preguntar_tipo:
                    contexto_extra = _marcar_respuesta_segura(
                        _respuesta_pregunta_tipo_producto(
                            texto_original=mensaje,
                            tipos_detectados=tipos_detectados,
                        )
                    )

                    nueva_etapa = "esperando_tipo_producto"
                    necesidad_ctx = _crear_ctx_tipo_producto(
                        texto_original=mensaje,
                        tipos_detectados=tipos_detectados,
                    )

                else:
                    # ------------------------------------------------
                    # Necesidad técnica estructurada
                    # ------------------------------------------------
                    # Si el cliente ya indicó dos o más campos técnicos
                    # explícitos, no debemos enviarlo al agente general
                    # de preguntas.
                    campos_declarados = extraer_campos_query(
                        mensaje
                    )

                    campos_significativos = {
                        nombre: valor
                        for nombre, valor in (
                            campos_declarados or {}
                        ).items()
                        if (
                            nombre != "valor_tecnico"
                            and str(valor or "").strip()
                        )
                    }

                    if len(campos_significativos) >= 2:
                        logger.info(
                            "Necesidad técnica estructurada detectada: "
                            "session=%s campos=%s",
                            session_id,
                            campos_significativos,
                        )

                        res = await buscar_en_catalogo(
                            mensaje
                        )

                        if debe_intentar_enriquecimiento(res):
                            res = await enriquecer_y_buscar(
                                mensaje
                            )

                        contexto_extra, nueva_etapa, necesidad_ctx = (
                            construir_respuesta_desde_resultado(
                                res=res,
                                cliente=cliente,
                                productos_acumulados=productos_acumulados,
                                desde="busqueda_tecnica_estructurada",
                                necesidad_ctx_base={
                                    "texto_original": mensaje,
                                    "query_evaluada": mensaje,
                                    "campos_declarados": campos_significativos,
                                },
                            )
                        )

                    else:
                        # --------------------------------------------
                        # Flujo tradicional
                        # --------------------------------------------
                        # Se mantiene para mensajes que todavía no
                        # contienen suficientes datos técnicos.
                        nec = await evaluar_necesidad(
                            mensaje
                        )

                        if nec["clara"]:
                            res = await buscar_en_catalogo(
                                mensaje
                            )

                            if debe_intentar_enriquecimiento(res):
                                res = await enriquecer_y_buscar(
                                    mensaje
                                )

                            contexto_extra, nueva_etapa, necesidad_ctx = (
                                construir_respuesta_desde_resultado(
                                    res=res,
                                    cliente=cliente,
                                    productos_acumulados=productos_acumulados,
                                    desde="busqueda",
                                    necesidad_ctx_base={
                                        "texto_original": mensaje,
                                        "query_evaluada": mensaje,
                                    },
                                )
                            )

                        else:
                            preguntas = _limpiar_preguntas_tecnicas(
                                nec.get("preguntas", [])
                            )

                            if preguntas:
                                necesidad_ctx = (
                                    _crear_ctx_preguntas_secuenciales(
                                        texto_original=mensaje,
                                        preguntas=preguntas,
                                        necesidad_ctx_base={
                                            "texto_original": mensaje,
                                            "dominio": nec.get("dominio"),
                                        },
                                    )
                                )

                                contexto_extra = _marcar_respuesta_segura(
                                    _respuesta_pregunta_tecnica_unica(
                                        preguntas[0],
                                        primera=True,
                                    )
                                )

                            else:
                                contexto_extra = _marcar_respuesta_segura(
                                    "Necesito un poco más de información "
                                    "para buscar la opción adecuada. "
                                    "¿Puedes indicarme la aplicación o "
                                    "especificación principal?"
                                )

                                necesidad_ctx = {
                                    "texto_original": mensaje
                                }

                            nueva_etapa = "descubrimiento"


        # Intenciones comerciales transversales
        # Solo se aplican si el estado comercial prioritario no resolvió el turno.
        # Esto evita que datos como cantidad, nombre, empresa o NIT sean tratados
        # como nuevas búsquedas o cambien de etapa por accidente.
        if True:
            if any(w in msg_lower for w in PALABRAS_MAS):
                nueva_etapa = "acumulando"

            elif any(w in msg_lower for w in PALABRAS_FIN):
                nueva_etapa = "cotizacion"

            elif "presupuesto" in msg_lower or "fecha" in msg_lower:
                nueva_etapa = "calificacion"

            elif "rut" in msg_lower or "proforma" in msg_lower:
                nueva_etapa = "proforma"

            elif "pago" in msg_lower or "pse" in msg_lower or "transferencia" in msg_lower:
                nueva_etapa = "pago"
    # ─────────────────────────────────────────────────────────
    # Construcción de contexto para LLM o respuesta segura
    # ─────────────────────────────────────────────────────────

    ctx_cliente = ""
    if cliente:
        partes = []
        if cliente.get("nombre"):
            partes.append(f"Nombre: {cliente['nombre']}")
        if cliente.get("empresa"):
            partes.append(f"Empresa: {cliente['empresa']}")
        if cliente.get("nit"):
            partes.append(f"NIT: {cliente['nit']}")
        if cliente.get("email"):
            partes.append(f"Email: {cliente['email']}")

        if partes:
            ctx_cliente = "[DATOS DEL CLIENTE]\n" + "\n".join(partes) + "\n"

    ctx_carrito = ""
    if productos_acumulados:
        ctx_carrito = f"[CARRITO: {len(productos_acumulados)} producto(s)]\n"

        for i, item in enumerate(productos_acumulados[-5:], start=1):
            prod = item.get("producto", {})
            ctx_carrito += f"{i}. {prod.get('codigo', '—')} | {prod.get('nombre', '—')}"

            if item.get("cantidad"):
                ctx_carrito += f" | cant: {item['cantidad']}"

            ctx_carrito += "\n"

    ctx_faltantes = ""
    if nueva_etapa in {"cotizacion", "calificacion"}:
        faltantes = datos_faltantes(cliente, "cotizacion")
        if faltantes:
            ctx_faltantes = f"[DATO FALTANTE — pregunta solo este: {faltantes[0]}]\n"

    elif nueva_etapa == "proforma":
        faltantes = datos_faltantes(cliente, "proforma")
        if faltantes:
            ctx_faltantes = f"[DATO FALTANTE — pregunta solo este: {faltantes[0]}]\n"

    system = PROMPT_MAESTRO
    partes_ctx = [
        c for c in [ctx_cliente, ctx_carrito, ctx_faltantes, contexto_extra]
        if c
    ]

    if partes_ctx:
        system += "\n\n---\nCONTEXTO ACTUAL:\n" + "\n".join(partes_ctx)

    msg_llm = mensaje if mensaje.strip() else f"[Cliente envió archivo: {archivo_nombre}]"

    respuesta_segura = _extraer_respuesta_segura(contexto_extra)

    if nueva_etapa == "cotizacion_lista" and not respuesta_segura:
        nombre_cliente = (cliente.get("nombre") or "").strip()

        if nombre_cliente:
            respuesta_segura = (
                f"Perfecto, {nombre_cliente}, ya quedé con tu solicitud. "
                "En breve recibirás la cotización en tu correo."
            )
        else:
            respuesta_segura = (
                "Perfecto, ya quedé con tu solicitud. "
                "En breve recibirás la cotización en tu correo."
            )

    if respuesta_segura:
        respuesta = respuesta_segura

    else:
        respuesta = await call_nia(
            system=system,
            historial=historial[-20:],
            mensaje_usuario=msg_llm,
        )

        if contiene_placeholder(respuesta):
            logger.warning(
                "Respuesta del LLM contenía placeholders. Se reemplaza por respuesta segura sin resultado."
            )
            respuesta = respuesta_sin_resultado(cliente=cliente)
            nueva_etapa = "descubrimiento"

    logger.info("Respuesta generada: etapa=%s session=%s", nueva_etapa, session_id)

    turno_user = {
        "role": "user",
        "content": msg_llm,
        "ts": datetime.utcnow().isoformat(),
    }

    turno_nia = {
        "role": "assistant",
        "content": respuesta,
        "ts": datetime.utcnow().isoformat(),
    }

    await _persistir_cliente_permanente(phone_id, cliente)

    await save_session(
        session_id=session_id,
        phone_id=phone_id,
        turnos=historial + [turno_user, turno_nia],
        etapa=nueva_etapa,
        archivo_activo=archivo_activo,
        necesidad_ctx=necesidad_ctx or {},
        cliente=cliente or {},
        productos_acumulados=productos_acumulados or [],
        cotizacion_recibida=cotizacion_recibida,
        archivo_cotizacion=archivo_cotizacion,
        proforma_recibida=proforma_recibida,
        archivo_proforma=archivo_proforma,
    )


    return {
        "respuesta": respuesta,
        "etapa": nueva_etapa,
        "items_resultado": items_resultado or None,
        "cliente": cliente or None,
    }


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────

@app.post("/nia/chat", response_model=ChatResponse)
@limiter.limit("30/minute")
async def nia_chat_texto(request: Request, req: ChatRequest):
    return ChatResponse(**await procesar_turno(
        session_id=req.session_id,
        mensaje=req.mensaje,
        phone_id=req.phone_id,
    ))


@app.post("/nia/chat/archivo", response_model=ChatResponse)
@limiter.limit("10/minute")
async def nia_chat_archivo(
    request: Request,
    session_id: str = Form(...),
    mensaje: str = Form(default=""),
    phone_id: str = Form(default=None),
    archivo: UploadFile = File(...),
):
    contenido = await archivo.read()

    return ChatResponse(**await procesar_turno(
        session_id=session_id,
        mensaje=mensaje,
        phone_id=phone_id,
        archivo_bytes=contenido,
        archivo_nombre=archivo.filename,
    ))


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "servicio": "NIA ViaIndustrial",
    }
