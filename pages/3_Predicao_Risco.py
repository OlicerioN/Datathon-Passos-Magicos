from __future__ import annotations

import io
import pandas as pd
import streamlit as st
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from ml_pipeline import calcular_risco_por_indicadores
from streamlit_utils import ensure_data_file

st.set_page_config(page_title="Predição de Risco do Aluno - Passos Mágicos", layout="wide")

ensure_data_file()

st.title("🎯 Predição de Risco do Aluno")
st.markdown("Avalie o risco de defasagem do aluno com base em seus indicadores")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Indicadores de Desempenho")
    iaa = st.slider("IAA (Autoavaliação)", 0.0, 10.0, 5.0, 0.1)
    ieg = st.slider("IEG (Engajamento)", 0.0, 10.0, 5.0, 0.1)

with col2:
    st.subheader("📈 Outros Indicadores")
    ips = st.slider("IPS (Indicador Psicossocial)", 0.0, 10.0, 5.0, 0.1)
    ida = st.slider("IDA (Desempenho Acadêmico)", 0.0, 10.0, 5.0, 0.1)

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    genero = st.selectbox("Gênero", ["Menina", "Menino", "Outro"], index=0)

with col2:
    instituicao = st.selectbox("Instituição de Ensino", ["Escola Publica", "Ensino Superior/Formado", "Escola Privada/Parceira", "Outros"], index=0)

with col3:
    idade = st.number_input("Idade", min_value=10, max_value=30, value=15)

st.divider()

if st.button("🔮 Prever Risco", use_container_width=True, key="predict_button"):
    probabilidade, nivel = calcular_risco_por_indicadores(iaa, ieg, ips, ida)
    
    st.session_state.predicao_resultado = {
        "probabilidade": probabilidade,
        "nivel": nivel,
        "iaa": iaa,
        "ieg": ieg,
        "ips": ips,
        "ida": ida,
        "genero": genero,
        "instituicao": instituicao,
        "idade": idade,
        "data": datetime.now().strftime("%d/%m/%Y %H:%M")
    }

if "predicao_resultado" in st.session_state:
    resultado = st.session_state.predicao_resultado
    probabilidade = resultado["probabilidade"]
    nivel = resultado["nivel"]
    
    percentual = probabilidade * 100
    
    if percentual < 30:
        cor_fundo = "#D4EDDA"
        cor_texto = "#155724"
        cor_borda = "#C3E6CB"
        emoji = "✅"
    elif percentual < 60:
        cor_fundo = "#FFF3CD"
        cor_texto = "#856404"
        cor_borda = "#FFEAA7"
        emoji = "⚠️"
    else:
        cor_fundo = "#F8D7DA"
        cor_texto = "#721C24"
        cor_borda = "#F5C6CB"
        emoji = "🚨"
    
    st.markdown(f"""
    <div style="background-color: {cor_fundo}; border-left: 5px solid {cor_borda}; padding: 20px; border-radius: 5px; margin: 20px 0;">
        <h2 style="color: {cor_texto}; margin-top: 0;">{emoji} Resultado da Predição</h2>
        <h1 style="color: {cor_texto}; font-size: 48px; margin: 10px 0;">{percentual:.1f}%</h1>
        <p style="color: {cor_texto}; font-size: 20px; margin: 5px 0;"><strong>Classificação: {nivel}</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("📋 Análise dos Fatores")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("IAA (Autoavaliação)", f"{resultado['iaa']:.1f}/10", 
                 delta=None)
        st.metric("IPS (Psicossocial)", f"{resultado['ips']:.1f}/10", 
                 delta=None)
    
    with col2:
        st.metric("IEG (Engajamento)", f"{resultado['ieg']:.1f}/10", 
                 delta=None)
        st.metric("IDA (Desempenho Acadêmico)", f"{resultado['ida']:.1f}/10", 
                 delta=None)
    
    st.divider()
    
    st.subheader("💡 Recomendações e Explicações")
    
    problemas = []
    recomendacoes = []
    
    if resultado['iaa'] < 5:
        problemas.append("Autoavaliação baixa: O aluno pode estar com baixa confiança em si mesmo")
        recomendacoes.append("Investir em atividades que aumentem a autoestima e confiança do aluno")
    
    if resultado['ieg'] < 5:
        problemas.append("Engajamento baixo: O aluno pode estar pouco envolvido com as atividades")
        recomendacoes.append("Aumentar o engajamento através de atividades dinâmicas e personalizadas")
    
    if resultado['ips'] < 5:
        problemas.append("Indicador psicossocial baixo: Possíveis problemas sociais ou emocionais")
        recomendacoes.append("Oferecer apoio psicossocial e acompanhamento especializado")
    
    if resultado['ida'] < 5:
        problemas.append("Desempenho acadêmico baixo: O aluno pode estar com dificuldade nas aprendizagens")
        recomendacoes.append("Implementar reforço escolar e acompanhamento pedagógico")
    
    if problemas:
        st.markdown("**Problemas Identificados:**")
        for problema in problemas:
            st.markdown(f"- {problema}")
    else:
        st.success("Nenhum problema específico identificado. Continue acompanhando o aluno!")
    
    if recomendacoes:
        st.markdown("**Recomendações:**")
        for i, rec in enumerate(recomendacoes, 1):
            st.markdown(f"{i}. {rec}")
    
    st.divider()
    
    st.subheader("📥 Baixar Relatório")
    
    def gerar_relatorio_pdf():
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
        
        elementos = []
        estilos = getSampleStyleSheet()
        
        titulo_style = ParagraphStyle(
            'CustomTitle',
            parent=estilos['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2E86AB'),
            spaceAfter=12,
            alignment=1
        )
        
        titulo = Paragraph("RELATÓRIO DE PREDIÇÃO DE RISCO", titulo_style)
        elementos.append(titulo)
        
        data_style = ParagraphStyle('DataStyle', parent=estilos['Normal'], fontSize=10, alignment=1)
        data_para = Paragraph(f"Gerado em: {resultado['data']}", data_style)
        elementos.append(data_para)
        elementos.append(Spacer(1, 12))
        
        elementos.append(Paragraph("<b>1. RESULTADO GERAL</b>", estilos['Heading2']))
        elementos.append(Spacer(1, 6))
        
        result_data = [
            ['Probabilidade de Risco', f"{percentual:.1f}%"],
            ['Classificação', nivel],
            ['Data da Avaliação', resultado['data']]
        ]
        
        result_table = Table(result_data, colWidths=[2.5*inch, 2.5*inch])
        result_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#E8F4F8')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elementos.append(result_table)
        elementos.append(Spacer(1, 12))
        
        elementos.append(Paragraph("<b>2. DADOS DO ALUNO</b>", estilos['Heading2']))
        elementos.append(Spacer(1, 6))
        
        aluno_data = [
            ['Idade', str(resultado['idade']) + ' anos'],
            ['Gênero', resultado['genero']],
            ['Instituição', resultado['instituicao']]
        ]
        
        aluno_table = Table(aluno_data, colWidths=[2.5*inch, 2.5*inch])
        aluno_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F0F0F0')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        elementos.append(aluno_table)
        elementos.append(Spacer(1, 12))
        
        elementos.append(Paragraph("<b>3. INDICADORES DE DESEMPENHO</b>", estilos['Heading2']))
        elementos.append(Spacer(1, 6))
        
        indicadores_data = [
            ['Indicador', 'Valor', 'Status'],
            ['IAA (Autoavaliação)', f"{resultado['iaa']:.1f}/10", 'Bom' if resultado['iaa'] >= 5 else 'Atenção'],
            ['IEG (Engajamento)', f"{resultado['ieg']:.1f}/10", 'Bom' if resultado['ieg'] >= 5 else 'Atenção'],
            ['IPS (Psicossocial)', f"{resultado['ips']:.1f}/10", 'Bom' if resultado['ips'] >= 5 else 'Atenção'],
            ['IDA (Desempenho Acadêmico)', f"{resultado['ida']:.1f}/10", 'Bom' if resultado['ida'] >= 5 else 'Atenção']
        ]
        
        ind_table = Table(indicadores_data, colWidths=[2.0*inch, 1.5*inch, 1.5*inch])
        ind_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F5F5F5')])
        ]))
        elementos.append(ind_table)
        elementos.append(Spacer(1, 12))
        
        elementos.append(Paragraph("<b>4. PROBLEMAS IDENTIFICADOS</b>", estilos['Heading2']))
        elementos.append(Spacer(1, 6))
        
        if problemas:
            for problema in problemas:
                elementos.append(Paragraph(f"• {problema}", estilos['Normal']))
                elementos.append(Spacer(1, 6))
        else:
            elementos.append(Paragraph("Nenhum problema específico identificado.", estilos['Normal']))
        
        elementos.append(Spacer(1, 12))
        
        elementos.append(Paragraph("<b>5. RECOMENDAÇÕES</b>", estilos['Heading2']))
        elementos.append(Spacer(1, 6))
        
        if recomendacoes:
            for i, rec in enumerate(recomendacoes, 1):
                elementos.append(Paragraph(f"{i}. {rec}", estilos['Normal']))
                elementos.append(Spacer(1, 6))
        else:
            elementos.append(Paragraph("Continue acompanhando o aluno conforme o cronograma regular.", estilos['Normal']))
        
        elementos.append(Spacer(1, 12))
        elementos.append(Paragraph("<i style='font-size: 8pt;'>Relatório confidencial. Informações destinadas apenas à equipe de acompanhamento do Passos Mágicos.</i>", estilos['Normal']))
        
        doc.build(elementos)
        buffer.seek(0)
        return buffer
    
    pdf_buffer = gerar_relatorio_pdf()
    nome_arquivo = f"relatorio_predicao_{datetime.now().strftime('%d%m%Y_%H%M%S')}.pdf"
    
    st.download_button(
        label="📄 Baixar Relatório em PDF",
        data=pdf_buffer.getvalue(),
        file_name=nome_arquivo,
        mime="application/pdf",
        use_container_width=True
    )
