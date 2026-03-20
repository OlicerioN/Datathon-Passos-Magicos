from __future__ import annotations
import streamlit as st
from streamlit_utils import ensure_data_file

st.set_page_config(page_title="Passos Mágicos - Solução Preditiva", layout="wide")

def main() -> None:
    ensure_data_file()

    st.title("Passos Mágicos - Solução Preditiva")
    st.markdown("""
                Plataforma analítica para acompanhar o risco de defasagem escolar e apoiar decisões
                pedagógicas com base em dados históricos do PEDE, fornecidos para realizar o Datathon Fase 5 FIAP.
                """)
    
    st.divider()

    st.subheader("O que você consegue analisar nesta aplicação", text_alignment='center')
    st.markdown("➤ Evolução do risco de defasagem ao longo dos anos. ㅤㅤ➤ Perfis com maior e menor risco para priorização de acompanhamento.", text_alignment='center')
    st.markdown("➤ Relação entre desempenho acadêmico e probabilidade de risco. ㅤㅤ➤ Diferenças entre grupos de alunos por gênero, instituição e nível de risco", text_alignment='center')

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Dashboard Geral")
        st.markdown(
            """
            Visão executiva para monitoramento rápido do cenário atual.

            **Nesta página você encontra:**
            - KPIs principais: total de alunos, taxa de risco e probabilidade média.
            - Distribuições por nível de risco e recortes por filtros.
            - Comparações de risco por ano, gênero e categoria de instituição.
            - Destaques de perfis com menor e maior risco para leitura prática.
            - Insights automáticos para apoio à tomada de decisão.

            **Use quando:**
            - Precisar de diagnóstico geral do programa.
            - Quiser priorizar grupos para intervenção imediata.
            """)
        
        with col2:
            st.markdown("### Análise de indicadores")
            st.markdown(
                """
                Visão analítica aprofundada dos indicadores pedagógicos e socioemocionais.

                **Nesta página você encontra:**
                - Foco exclusivo em IAA, IEG, IPS e IDA.
                - Evolução temporal dos indicadores por ano-base.
                - Comparação entre alunos com e sem risco de defasagem.
                - Distribuição, dispersão e estabilidade dos indicadores.

                **Use quando:**
                - Quiser entender causas e padrões por trás do risco.
                - Precisar embasar ações pedagógicas por indicador específico.
                """)
            
    st.divider()

    st.info("Navegue pelo menu lateral para acessar as visões principais do projeto.")
    st.caption("Recomendação de uso: comece pelo Dashboard Geral para visão macro e avance para Análise de Indicadores para investigação detalhada.")

if __name__ == "__main__":
    main()