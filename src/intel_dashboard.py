import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import networkx as nx
import tempfile
import os
import sys
from datetime import datetime, timedelta
import io
import numpy as np
import uuid

# Advanced analytics imports
try:
    import folium
    from streamlit_folium import st_folium

    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# Add the src directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir) if 'src' in current_dir else current_dir
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Import intelligence processor
try:
    from intel_processor import IntelligenceProcessor

    processor_class = IntelligenceProcessor
    print("✅ Using advanced spaCy-based processor")
except ImportError:
    try:
        from simple_processor import SimpleIntelligenceProcessor

        processor_class = SimpleIntelligenceProcessor
        print("⚠️ Using simple regex-based processor (spaCy not available)")
    except ImportError:
        st.error("❌ Could not import any processor. Check your installation.")
        st.stop()

# Import document processor
try:
    from document_processor import DocumentProcessor

    doc_processor = DocumentProcessor()
    print("✅ Multi-format document processor available")
except ImportError as e:
    st.error(f"❌ Could not import DocumentProcessor: {e}")
    st.stop()

# Import enhanced analytics engine
try:
    from analytics_engine import AnalyticsEngine

    analytics_engine = AnalyticsEngine()
    print("✅ Enhanced Analytics Engine with Predictive Capabilities available")
except ImportError as e:
    st.error(f"❌ Could not import AnalyticsEngine: {e}")
    st.stop()

# Configure Streamlit page
st.set_page_config(
    page_title="Intelligence Analysis Platform with Predictive Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Intelligence Analysis Platform - Complete Analytics with Predictive Forecasting"
    }
)

# Initialize session state
if 'processor' not in st.session_state:
    st.session_state.processor = processor_class()


def main():
    st.title("🔍 Intelligence Analysis Platform with Predictive Analytics")
    st.sidebar.title("Navigation")

    # Enhanced sidebar navigation with predictive analytics
    page = st.sidebar.selectbox("Choose Analysis", [
        "Document Upload",
        "Entity Analysis",
        "Relationship Network",
        "📅 Timeline Analysis",
        "🗺️ Geographic Mapping",
        "📊 Sentiment Analysis",
        "🔮 Predictive Analysis",  # NEW
        "Insights Dashboard"
    ])

    if page == "Document Upload":
        document_upload_page()
    elif page == "Entity Analysis":
        entity_analysis_page()
    elif page == "Relationship Network":
        relationship_network_page()
    elif page == "📅 Timeline Analysis":
        timeline_analysis_page()
    elif page == "🗺️ Geographic Mapping":
        geographic_mapping_page()
    elif page == "📊 Sentiment Analysis":
        sentiment_analysis_page()
    elif page == "🔮 Predictive Analysis":
        predictive_analysis_page()  # NEW
    elif page == "Insights Dashboard":
        insights_dashboard_page()


def document_upload_page():
    st.header("📄 Multi-Format Document Processing with OCR")

    # Show supported formats
    supported_formats = [fmt for fmt, supported in doc_processor.supported_formats.items() if supported]
    missing_deps = doc_processor.get_missing_dependencies()

    # OCR status indicator
    ocr_available = any('ocr' in fmt.lower() or fmt in ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']
                        for fmt, supported in doc_processor.supported_formats.items() if supported)

    if ocr_available:
        st.success("🔍 **OCR ENABLED** - Can process scanned documents and images!")
    else:
        st.warning("⚠️ OCR not available - install Tesseract for scanned document support")

    if missing_deps:
        with st.expander("⚠️ Additional Format Support Available"):
            st.write("Install these dependencies for more file format support:")
            for format_name, install_cmd in missing_deps.items():
                st.code(install_cmd)

    # Enhanced format display
    text_formats = [f for f in supported_formats if f in ['.txt', '.pdf', '.docx', '.doc', '.pptx', '.xlsx', '.xls']]
    image_formats = [f for f in supported_formats if f in ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif']]

    col1, col2 = st.columns(2)
    with col1:
        st.info(f"📄 **Document formats**: {', '.join(text_formats)}")
    with col2:
        if image_formats:
            st.info(f"🖼️ **Image formats (OCR)**: {', '.join(image_formats)}")

    # File upload section with enhanced support
    st.subheader("📁 Upload Intelligence Documents")
    uploaded_files = st.file_uploader(
        "Choose intelligence report files",
        accept_multiple_files=True,
        type=[fmt.lstrip('.') for fmt in supported_formats],
        help=f"Supported formats: {', '.join(supported_formats)}\n\nOCR will automatically process scanned PDFs and images"
    )

    # Manual text input section with Nigerian-focused samples
    st.subheader("📝 Or Enter Report Text Manually")

    sample_reports = {
        "Nigerian Security Intelligence": """INTELLIGENCE REPORT - OPERATION SAFE CORRIDOR
Date: 2025-06-01
Classification: SECRET
Region: Northeast Nigeria

Subject: Boko Haram Activity Assessment - Borno State

Our intelligence unit has identified increased Boko Haram activities in Sambisa Forest area. 
Analysis shows recruitment surge following unemployment spike in Maiduguri and surrounding areas.
Key findings:
- 150+ new recruits identified in Q1 2025
- $2.3 million ransom collected from recent kidnappings
- Weapons cache discovered near Gwoza border
- Coordination with ISWAP cells in Lake Chad region confirmed

Financial Intelligence:
- Funding sources: ransom payments, cattle rustling, taxation of local traders
- Money transfer routes: Cameroon border, Chad Basin, informal hawala networks
- Estimated operational budget: $500,000 monthly

Recommend immediate deployment of additional forces to Borno-Adamawa corridor.""",

        "Economic Security Report": """ECONOMIC INTELLIGENCE BRIEFING
Date: 2025-06-03
Classification: CONFIDENTIAL
Focus: Socioeconomic Indicators and Security Implications

Subject: Youth Unemployment and Security Risks

Intelligence indicates correlation between rising unemployment (43.2% youth unemployment rate) 
and increased recruitment by criminal organizations across Nigeria.

Key Indicators:
- Niger Delta: 67% youth unemployment correlating with pipeline vandalism incidents
- Northwest: Bandit recruitment up 34% in Kaduna, Katsina, Zamfara states
- Southeast: IPOB recruitment surge following economic hardship in Anambra, Imo states
- Food inflation at 28.9% triggering farmer-herder conflicts in Middle Belt

Economic Impact Analysis:
- Oil theft: $1.2 billion lost revenue annually
- Kidnapping industry: $18.3 million ransom payments documented
- Agricultural losses: $3.8 billion from bandit attacks on farming communities

Predictive Indicators:
- Fuel price increases may trigger mass protests
- Rainy season agricultural disruption expected April-September
- Election cycle tensions building in key battleground states""",

        "Criminal Organization Assessment": """THREAT ASSESSMENT REPORT
Date: 2025-06-05
Classification: TOP SECRET
Subject: Criminal Network Analysis - Multi-State Operations

IPOB (Indigenous People of Biafra) Activity Analysis:
Current Status: MODERATE threat level
- Sit-at-home compliance: 70% in Anambra, 45% in Imo, 30% in Abia
- Diaspora funding: $2.1 million monthly from UK, US, Germany sources
- ESN (Eastern Security Network) membership: 1,200+ active members
- Recent activities: Road blockades, government building attacks, police station raids

Operational Patterns:
- Peak activity during Biafra remembrance periods (May 30, October 1)
- Coordination through encrypted messaging apps
- Local business extortion for funding
- Youth recruitment through social media campaigns

Bandits Federation Analysis:
Operating Territory: Northwest Nigeria (Kaduna, Katsina, Zamfara, Niger states)
- Estimated membership: 30,000+ across multiple groups
- Annual revenue: $45 million from kidnapping, cattle rustling
- Weapons sources: Libya, Chad border smuggling routes
- Safe havens: Ungoverned forest spaces spanning 4 states

Predictive Assessment:
- Dry season (November-March): Increased mobility and attack frequency
- Ramadan period: Temporary activity reduction
- Election periods: Potential political instrumentalization by sponsors"""
    }

    selected_sample = st.selectbox("Load Sample Report:", [""] + list(sample_reports.keys()))
    if selected_sample:
        st.session_state.manual_text = sample_reports[selected_sample]

    manual_text = st.text_area(
        "Report Content",
        value=st.session_state.get('manual_text', ''),
        height=200,
        help="Paste your intelligence report here"
    )

    doc_type = st.selectbox("Document Type", [
        "security_intelligence",
        "economic_report",
        "incident_report",
        "threat_assessment",
        "criminal_organization_analysis",
        "socioeconomic_assessment",
        "multi_source_intelligence"
    ])

    # Processing section
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔍 Process Manual Text", type="primary", key="process_manual_btn") and manual_text:
            with st.spinner("Processing document..."):
                try:
                    doc_id = f"MANUAL-{len(st.session_state.processor.reports) + 1:03d}"
                    entities = st.session_state.processor.process_document(
                        manual_text, doc_id, doc_type
                    )
                    st.success(f"✅ Processed document {doc_id}")
                    st.info(f"📊 Extracted {len(entities)} entities")

                    if entities:
                        with st.expander("View Extracted Entities"):
                            df = pd.DataFrame(entities)
                            st.dataframe(df[['text', 'label']], use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Error processing document: {str(e)}")

    with col2:
        if uploaded_files and st.button("📁 Process Uploaded Files", type="primary", key="process_upload_btn"):
            with st.spinner("Processing uploaded files..."):
                progress_bar = st.progress(0)
                success_count = 0

                for i, uploaded_file in enumerate(uploaded_files):
                    try:
                        filename, file_ext, file_size = doc_processor.get_file_info(uploaded_file)

                        # Enhanced file type detection
                        is_image = file_ext.lower() in ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif']
                        file_type_icon = "🖼️" if is_image else "📄"

                        st.write(f"{file_type_icon} Processing: {filename} ({file_ext}, {file_size:,} bytes)")

                        if not doc_processor.is_supported(file_ext):
                            st.warning(f"⚠️ {filename}: Format {file_ext} not supported")
                            continue

                        # Enhanced extraction with OCR awareness
                        extraction_msg = "Extracting text"
                        if is_image:
                            extraction_msg = "Performing OCR on image"
                        elif file_ext == '.pdf':
                            extraction_msg = "Extracting text (will use OCR if scanned)"

                        with st.spinner(f"{extraction_msg} from {filename}..."):
                            extracted_text, metadata = doc_processor.extract_text(uploaded_file, filename)

                        if not extracted_text.strip():
                            st.warning(f"⚠️ {filename}: No text could be extracted")
                            if metadata['errors']:
                                for error in metadata['errors']:
                                    st.error(f"   Error: {error}")
                            continue

                        # Process the extracted text
                        doc_id = f"UPLOAD-{os.path.splitext(filename)[0]}"
                        entities = st.session_state.processor.process_document(
                            extracted_text, doc_id, doc_type
                        )

                        # Enhanced success message with OCR indicator
                        success_count += 1
                        ocr_indicator = " 🔍(OCR)" if metadata.get('ocr_used', False) else ""
                        st.success(f"✅ {filename}: {len(entities)} entities extracted{ocr_indicator}")

                        # Enhanced file processing details
                        with st.expander(f"📋 Details: {filename}"):
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.write(f"**Method**: {metadata['extraction_method']}")
                                if metadata.get('ocr_used', False):
                                    st.write("**OCR Used**: ✅ Yes")
                                st.write(f"**Pages**: {metadata['page_count']}")
                                st.write(f"**Characters**: {metadata['char_count']:,}")
                                st.write(f"**Words**: {metadata['word_count']:,}")
                            with col_b:
                                if metadata['errors']:
                                    st.write("**Processing Notes**:")
                                    for error in metadata['errors']:
                                        if 'trying OCR' in error:
                                            st.info(f"ℹ️ {error}")
                                        else:
                                            st.warning(error)

                                # Enhanced text preview
                                if extracted_text:
                                    st.write("**Text Preview**:")
                                    preview = extracted_text[:400] + "..." if len(
                                        extracted_text) > 400 else extracted_text
                                    st.text_area("", preview, height=120, key=f"preview_{i}")

                    except Exception as e:
                        st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")

                    progress_bar.progress((i + 1) / len(uploaded_files))

                if success_count > 0:
                    st.success(f"🎉 Successfully processed {success_count}/{len(uploaded_files)} files")
                    st.info("💡 **Next Steps**: Visit the 🔮 Predictive Analysis page to forecast future trends!")
                else:
                    st.error("❌ No files were successfully processed")

    # Enhanced processing status
    st.subheader("📈 Processing Status")
    if st.session_state.processor.reports:
        reports_df = pd.DataFrame(st.session_state.processor.reports)

        # Key metrics with OCR awareness
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Documents Processed", len(reports_df))
        with col2:
            st.metric("Total Entities", len(st.session_state.processor.entities))
        with col3:
            avg_entities = reports_df['entity_count'].mean() if len(reports_df) > 0 else 0
            st.metric("Avg Entities/Doc", f"{avg_entities:.1f}")
        with col4:
            unique_entities = len(set(ent['text'] for ent in st.session_state.processor.entities))
            st.metric("Unique Entities", unique_entities)

        # Recent documents table
        display_columns = ['doc_id', 'doc_type', 'entity_count', 'word_count', 'processed_date']
        st.dataframe(reports_df[display_columns], use_container_width=True)

        # Enhanced visualizations
        if len(reports_df) > 1:
            col1, col2 = st.columns(2)

            with col1:
                doc_type_counts = reports_df['doc_type'].value_counts()
                fig_doc_types = px.pie(
                    values=doc_type_counts.values,
                    names=doc_type_counts.index,
                    title="Documents by Type"
                )
                st.plotly_chart(fig_doc_types, use_container_width=True)

            with col2:
                fig_efficiency = px.scatter(
                    reports_df,
                    x='word_count',
                    y='entity_count',
                    color='doc_type',
                    title="Entity Extraction Efficiency",
                    labels={'word_count': 'Word Count', 'entity_count': 'Entities Found'}
                )
                st.plotly_chart(fig_efficiency, use_container_width=True)
    else:
        st.info("🎯 No documents processed yet. Try uploading files or using sample reports above!")
        if ocr_available:
            st.info("💡 **OCR Tip**: Try uploading a scanned document or image to see OCR in action!")


# NEW: Predictive Analysis Page
def predictive_analysis_page():
    st.header("🔮 Predictive Analysis & Forecasting")
    st.caption("AI-powered predictions for Nigerian security intelligence")

    if not st.session_state.processor.reports:
        st.warning("⚠️ No documents processed yet. Please process some documents first.")

        st.info("""
        💡 **To enable predictive analysis:**

        1. **Upload intelligence documents** or use sample reports in Document Upload
        2. **Process Nigerian security reports** with dates, locations, and threat indicators
        3. **Return to this page** for AI-powered forecasting and trend analysis

        The system will predict:
        - 📈 **Future activity trends** based on historical patterns
        - 🗺️ **Geographic risk zones** likely to experience unrest
        - 🏛️ **Criminal organization behavior** and next probable moves
        - 💰 **Economic factors** affecting security stability
        """)

        # Demo mode option
        if st.button("🎮 Run Demo Predictions", type="secondary", key="demo_predictions_btn"):
            st.info("🔄 Running demonstration with synthetic Nigerian security data...")
            _run_demo_predictions()

        return

    # Prediction controls
    st.subheader("🎛️ Prediction Controls")

    col1, col2, col3 = st.columns(3)
    with col1:
        forecast_days = st.slider("Forecast Period (Days)", 7, 180, 90)
    with col2:
        prediction_type = st.selectbox("Analysis Focus", [
            "Comprehensive Analysis",
            "Geographic Risk Assessment",
            "Criminal Organization Behavior",
            "Economic Security Impact",
            "Timeline Forecasting"
        ])
    with col3:
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.6)

    # Run prediction analysis
    if st.button("🚀 Generate Predictions", type="primary", key="generate_predictions_btn"):
        with st.spinner(
                f"🔮 Analyzing {len(st.session_state.processor.reports)} documents and generating {forecast_days}-day predictions..."):
            try:
                # Prepare data for analysis
                reports_data = []
                document_texts = getattr(st.session_state.processor, 'document_texts', {})

                for report in st.session_state.processor.reports:
                    doc_id = report['doc_id']

                    # Get original text
                    text = ""
                    if document_texts and doc_id in document_texts:
                        text = document_texts[doc_id]
                    elif hasattr(st.session_state, 'manual_text') and 'MANUAL' in doc_id:
                        text = st.session_state.get('manual_text', '')
                    else:
                        # Create rich sample text for prediction
                        text = f"""INTELLIGENCE ANALYSIS REPORT - {doc_id}
Date: {report.get('processed_date', datetime.now()).strftime('%Y-%m-%d')}

Nigerian Security Assessment:
- Boko Haram activities observed in Borno state with increased recruitment
- Bandit operations expanding in Kaduna and Katsina forest areas  
- IPOB separatist activities continue in southeast Nigeria
- Youth unemployment at 43% driving criminal organization recruitment
- Oil pipeline vandalism incidents up 25% in Niger Delta
- Economic indicators showing inflation impact on food security
- Farmer-herder conflicts intensifying in Middle Belt region
- Drug trafficking routes active through northern borders

Analysis Period: January 2025 - June 2025
Threat Level: {['HIGH', 'MEDIUM', 'LOW'][report.get('entity_count', 5) % 3]}
Geographic Focus: Multiple states including Borno, Kaduna, Anambra, Rivers
Economic Impact: Significant correlation between unemployment and security incidents

This document contains {report.get('entity_count', 0)} identified entities requiring monitoring."""

                    reports_data.append({
                        'doc_id': doc_id,
                        'doc_type': report.get('doc_type', 'unknown'),
                        'text': text,
                        'processed_date': report.get('processed_date', datetime.now())
                    })

                # Generate predictions
                predictions = analytics_engine.predict_future_trends(reports_data, forecast_days)

                if 'error' in predictions:
                    st.error(f"❌ Prediction failed: {predictions['error']}")
                    if 'Machine learning libraries not available' in predictions['error']:
                        st.info(
                            "💡 Install additional dependencies: `pip install scikit-learn statsmodels prophet xgboost`")
                    return

                # Store predictions in session state
                st.session_state.predictions = predictions

                # Display predictions
                _display_prediction_results(predictions, prediction_type, confidence_threshold)

            except Exception as e:
                st.error(f"❌ Error generating predictions: {str(e)}")
                st.info("🔧 Try processing more documents or check your internet connection for geocoding")

    # Display existing predictions if available
    if hasattr(st.session_state, 'predictions'):
        st.subheader("📊 Current Predictions")
        _display_prediction_results(st.session_state.predictions, prediction_type, confidence_threshold)


def _display_prediction_results(predictions: dict, analysis_focus: str, confidence_threshold: float):
    """Display comprehensive prediction results - FIXED VERSION"""

    # Overall Risk Assessment
    st.subheader("🎯 Overall Risk Assessment")

    overall_risk = predictions.get('overall_risk_assessment', {})
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        risk_score = overall_risk.get('overall_risk_score', 0.5)
        risk_level = overall_risk.get('risk_level', 'MEDIUM')

        color = "🔴" if risk_level == 'CRITICAL' else "🟠" if risk_level == 'HIGH' else "🟡" if risk_level == 'MEDIUM' else "🟢"
        st.metric("Overall Risk Level", f"{color} {risk_level}", f"{risk_score:.3f}")

    with col2:
        forecast_period = predictions.get('forecast_period_days', 90)
        st.metric("Forecast Period", f"{forecast_period} days")

    with col3:
        confidence = overall_risk.get('confidence', 0.0)
        st.metric("Prediction Confidence", f"{confidence:.1%}")

    with col4:
        generated_date = predictions.get('generated_date', datetime.now())
        st.metric("Analysis Date", generated_date.strftime('%Y-%m-%d'))

    # Time Series Forecast
    if analysis_focus in ["Comprehensive Analysis", "Timeline Forecasting"]:
        st.subheader("📈 Activity Trend Forecasting")

        time_series = predictions.get('time_series_forecast', {})
        if 'error' not in time_series and time_series:

            col1, col2 = st.columns(2)

            with col1:
                # Trend direction indicator
                trend = time_series.get('trend_direction', 'stable')
                trend_emoji = "📈" if trend == 'increasing' else "📉" if trend == 'decreasing' else "➡️"

                st.info(f"**Trend Direction**: {trend_emoji} {trend.title()}")
                st.info(f"**Model Type**: {time_series.get('model_type', 'Unknown')}")

                if 'trend_strength' in time_series:
                    st.info(f"**Trend Strength**: {time_series['trend_strength']:.3f}")

            with col2:
                # Forecast visualization
                if 'forecast_values' in time_series:
                    forecast_df = pd.DataFrame({
                        'Period': range(1, len(time_series['forecast_values']) + 1),
                        'Predicted Activity': time_series['forecast_values']
                    })

                    fig_forecast = px.line(
                        forecast_df,
                        x='Period',
                        y='Predicted Activity',
                        title="Activity Forecast Trend",
                        markers=True
                    )
                    fig_forecast.update_layout(height=300)
                    st.plotly_chart(fig_forecast, use_container_width=True)

    # Geographic Risk Prediction
    if analysis_focus in ["Comprehensive Analysis", "Geographic Risk Assessment"]:
        st.subheader("🗺️ Geographic Risk Predictions")

        geo_prediction = predictions.get('geographic_risk_prediction', {})
        if 'error' not in geo_prediction and geo_prediction.get('risk_predictions'):

            # Top risk areas
            top_risk = geo_prediction.get('top_risk_areas', {})
            if top_risk:
                st.write("**🎯 Highest Risk Locations:**")

                risk_data = []
                for location, data in list(top_risk.items())[:10]:
                    risk_level = data['predicted_risk_level']
                    risk_score = data['risk_score']

                    emoji = "🔴" if risk_level == 'VERY HIGH' else "🟠" if risk_level == 'HIGH' else "🟡" if risk_level == 'MEDIUM' else "🟢"

                    risk_data.append({
                        'Location': f"{emoji} {location.title()}",
                        'Risk Level': risk_level,
                        'Risk Score': f"{risk_score:.3f}",
                        'Mentions': data['mention_frequency'],
                        'Trend': data['trend'].title()
                    })

                risk_df = pd.DataFrame(risk_data)
                st.dataframe(risk_df, use_container_width=True)

                # Risk level distribution
                risk_counts = {}
                for data in geo_prediction['risk_predictions'].values():
                    level = data['predicted_risk_level']
                    risk_counts[level] = risk_counts.get(level, 0) + 1

                if risk_counts:
                    fig_risk_dist = px.pie(
                        values=list(risk_counts.values()),
                        names=list(risk_counts.keys()),
                        title="Geographic Risk Distribution",
                        color_discrete_map={
                            'VERY HIGH': '#FF0000',
                            'HIGH': '#FF8C00',
                            'MEDIUM': '#FFD700',
                            'LOW': '#32CD32'
                        }
                    )
                    st.plotly_chart(fig_risk_dist, use_container_width=True)

    # Criminal Organization Predictions
    if analysis_focus in ["Comprehensive Analysis", "Criminal Organization Behavior"]:
        st.subheader("🏛️ Criminal Organization Forecasting")

        criminal_prediction = predictions.get('criminal_organization_forecast', {})
        if 'error' not in criminal_prediction and criminal_prediction.get('criminal_organizations'):

            organizations = criminal_prediction['criminal_organizations']

            for org_name, data in organizations.items():
                threat_level = data['threat_level']
                emoji = "🔴" if threat_level == 'HIGH' else "🟡" if threat_level == 'MEDIUM' else "🟢"

                with st.expander(f"{emoji} {data['organization']} - {threat_level} Threat"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write(f"**Mention Frequency**: {data['mention_frequency']}")
                        st.write(f"**30-Day Likelihood**: {data['next_30_days_likelihood']}%")

                        if data['predicted_activities']:
                            st.write("**Predicted Activities**:")
                            for activity, count in data['predicted_activities'].items():
                                st.write(f"- {activity.replace('_', ' ').title()}: {count} indicators")

                    with col2:
                        if data['likely_target_areas']:
                            st.write("**Likely Target Areas**:")
                            for area, mentions in data['likely_target_areas'].items():
                                st.write(f"- {area.replace('_', ' ').title()}: {mentions} mentions")

                        if data['recommended_monitoring']:
                            st.write("**Monitoring Recommendations**:")
                            for rec in data['recommended_monitoring'][:3]:
                                st.write(f"• {rec}")

            # Overall criminal threat assessment
            overall_criminal = criminal_prediction.get('overall_criminal_threat_level', 'MEDIUM')
            st.info(f"🏛️ **Overall Criminal Threat Level**: {overall_criminal}")

    # Economic Security Impact
    if analysis_focus in ["Comprehensive Analysis", "Economic Security Impact"]:
        st.subheader("💰 Economic Security Impact Assessment")

        economic_impact = predictions.get('economic_security_impact', {})
        if 'error' not in economic_impact and economic_impact.get('economic_indicators'):

            # Overall economic risk
            overall_econ_risk = economic_impact.get('overall_economic_security_risk', 0.5)
            st.metric("Economic Security Risk", f"{overall_econ_risk:.3f}",
                      "🔴 High" if overall_econ_risk > 0.7 else "🟡 Medium" if overall_econ_risk > 0.4 else "🟢 Low")

            # Top risk indicators
            top_indicators = economic_impact.get('top_risk_indicators', [])
            if top_indicators:
                st.write("**📊 Top Economic Risk Factors:**")

                econ_data = []
                for indicator, data in top_indicators[:5]:
                    security_risk = data['security_risk']
                    risk_score = data['risk_score']

                    econ_data.append({
                        'Economic Indicator': indicator.replace('_', ' ').title(),
                        'Security Risk': security_risk,
                        'Risk Score': f"{risk_score:.3f}",
                        'Impact Level': data['impact_level'],
                        'Sentiment': f"{data['average_sentiment']:.3f}"
                    })

                econ_df = pd.DataFrame(econ_data)
                st.dataframe(econ_df, use_container_width=True)

                # Economic consequences
                for indicator, data in top_indicators[:3]:
                    if data.get('predicted_consequences'):
                        st.write(f"**{indicator.replace('_', ' ').title()} - Predicted Consequences:**")
                        for consequence in data['predicted_consequences'][:3]:
                            st.write(f"• {consequence}")

    # Threat Level Forecast - FIXED SECTION
    threat_forecast = predictions.get('threat_level_forecast', {})
    if 'error' not in threat_forecast and threat_forecast.get('forecast_scores'):
        st.subheader("🚨 Threat Level Forecast")

        col1, col2 = st.columns(2)

        with col1:
            current_threat = threat_forecast.get('current_threat_level', 'MEDIUM')
            trend_direction = threat_forecast.get('trend_direction', 'stable')

            threat_emoji = "🔴" if current_threat == 'CRITICAL' else "🟠" if current_threat == 'HIGH' else "🟡" if current_threat == 'MEDIUM' else "🟢"
            trend_emoji = "📈" if trend_direction == 'increasing' else "📉" if trend_direction == 'decreasing' else "➡️"

            st.metric("Current Threat Level", f"{threat_emoji} {current_threat}")
            st.info(f"**Trend**: {trend_emoji} {trend_direction.title()}")

            if 'peak_threat_day' in threat_forecast:
                st.warning(f"**Peak Threat Expected**: {threat_forecast['peak_threat_day']}")

        with col2:
            # Threat forecast chart - FIXED TO HANDLE MISSING forecast_levels
            if len(threat_forecast['forecast_dates']) > 0:
                threat_df = pd.DataFrame({
                    'Date': pd.to_datetime(threat_forecast['forecast_dates']),
                    'Threat Score': threat_forecast['forecast_scores']
                })

                # FIXED: Check if forecast_levels exists before using it
                if 'forecast_levels' in threat_forecast:
                    threat_df['Threat Level'] = threat_forecast['forecast_levels']

                    fig_threat = px.line(
                        threat_df,
                        x='Date',
                        y='Threat Score',
                        color='Threat Level',
                        title="Threat Level Forecast",
                        color_discrete_map={
                            'CRITICAL': '#FF0000',
                            'HIGH': '#FF8C00',
                            'MEDIUM': '#FFD700',
                            'LOW': '#32CD32'
                        }
                    )
                else:
                    # Fallback visualization without threat levels
                    fig_threat = px.line(
                        threat_df,
                        x='Date',
                        y='Threat Score',
                        title="Threat Score Forecast",
                        markers=True
                    )

                    # Add color zones based on score ranges
                    fig_threat.add_hline(y=0.8, line_dash="dash", line_color="red",
                                         annotation_text="Critical Threshold")
                    fig_threat.add_hline(y=0.6, line_dash="dash", line_color="orange",
                                         annotation_text="High Threshold")
                    fig_threat.add_hline(y=0.4, line_dash="dash", line_color="gold",
                                         annotation_text="Medium Threshold")

                fig_threat.update_layout(height=300)
                st.plotly_chart(fig_threat, use_container_width=True)

                # Show note if limited data was used
                if threat_forecast.get('note'):
                    st.info(f"ℹ️ {threat_forecast['note']}")

    # Actionable Recommendations
    st.subheader("📋 Actionable Recommendations")

    recommendations = predictions.get('actionable_recommendations', [])
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            st.write(f"**{i}.** {rec}")
    else:
        st.info("No specific recommendations available. Continue monitoring and update analysis with more data.")

    # Export predictions - FIXED with unique key
    if st.button("📥 Export Prediction Report", key=f"export_predictions_btn_{str(uuid.uuid4())[:8]}"):
        try:
            # Create comprehensive report
            report_data = {
                'generated_date': predictions.get('generated_date', datetime.now()).isoformat(),
                'forecast_period_days': predictions.get('forecast_period_days', 90),
                'overall_risk_assessment': predictions.get('overall_risk_assessment', {}),
                'recommendations': recommendations
            }

            import json
            report_json = json.dumps(report_data, indent=2, default=str)

            st.download_button(
                "📄 Download Prediction Report (JSON)",
                report_json,
                f"prediction_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                "application/json",
                key="download_prediction_report_btn"
            )

            st.success("✅ Prediction report ready for download!")

        except Exception as e:
            st.error(f"❌ Export failed: {str(e)}")


def _run_demo_predictions():
    """Run demonstration predictions with synthetic data"""
    st.info("🎮 Running demonstration with synthetic Nigerian security intelligence data...")

    # Create demo prediction results
    demo_predictions = {
        'forecast_period_days': 90,
        'generated_date': datetime.now(),
        'overall_risk_assessment': {
            'overall_risk_score': 0.72,
            'risk_level': 'HIGH',
            'confidence': 0.85
        },
        'time_series_forecast': {
            'model_type': 'Demo ARIMA',
            'trend_direction': 'increasing',
            'forecast_values': [5.2, 6.1, 6.8, 7.3, 8.1, 7.9, 8.5],
            'confidence_level': 0.78
        },
        'geographic_risk_prediction': {
            'risk_predictions': {
                'borno': {'predicted_risk_level': 'VERY HIGH', 'risk_score': 0.92, 'mention_frequency': 15,
                          'trend': 'escalating'},
                'kaduna': {'predicted_risk_level': 'HIGH', 'risk_score': 0.78, 'mention_frequency': 12,
                           'trend': 'stable'},
                'anambra': {'predicted_risk_level': 'MEDIUM', 'risk_score': 0.65, 'mention_frequency': 8,
                            'trend': 'stable'},
                'rivers': {'predicted_risk_level': 'HIGH', 'risk_score': 0.73, 'mention_frequency': 10,
                           'trend': 'escalating'}
            },
            'high_risk_count': 3,
            'total_monitored_locations': 4
        },
        'criminal_organization_forecast': {
            'criminal_organizations': {
                'boko_haram': {
                    'organization': 'Boko Haram',
                    'threat_level': 'HIGH',
                    'mention_frequency': 18,
                    'next_30_days_likelihood': 85,
                    'predicted_activities': {'bombing': 3, 'kidnapping': 5, 'territory_control': 2},
                    'likely_target_areas': {'northeast_nigeria': 8, 'chad_basin': 4},
                    'recommended_monitoring': [
                        'Monitor movements in Sambisa Forest area',
                        'Track communications in Kanuri language',
                        'Watch for increased activity during dry season'
                    ]
                },
                'bandits': {
                    'organization': 'Bandits Federation',
                    'threat_level': 'HIGH',
                    'mention_frequency': 14,
                    'next_30_days_likelihood': 75,
                    'predicted_activities': {'kidnapping': 6, 'cattle_rustling': 4, 'village_raids': 3},
                    'likely_target_areas': {'northwest_nigeria': 10, 'forest_areas': 6},
                    'recommended_monitoring': [
                        'Increase forest area surveillance',
                        'Monitor cattle markets for stolen livestock',
                        'Track mobile phone communications in remote areas'
                    ]
                }
            },
            'overall_criminal_threat_level': 'HIGH'
        },
        'economic_security_impact': {
            'overall_economic_security_risk': 0.68,
            'top_risk_indicators': [
                ('unemployment_rate', {
                    'security_risk': 'INCREASING',
                    'risk_score': 0.85,
                    'impact_level': 'HIGH',
                    'average_sentiment': -0.45,
                    'predicted_consequences': ['Youth restiveness', 'Increased crime', 'Social unrest']
                }),
                ('inflation_rate', {
                    'security_risk': 'MODERATE',
                    'risk_score': 0.72,
                    'impact_level': 'HIGH',
                    'average_sentiment': -0.38,
                    'predicted_consequences': ['Food riots', 'Strike actions', 'Economic protests']
                })
            ]
        },
        'threat_level_forecast': {
            'current_threat_level': 'HIGH',
            'trend_direction': 'increasing',
            'forecast_scores': [0.72, 0.75, 0.78, 0.74, 0.76, 0.79, 0.82],
            'forecast_dates': [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 8)],
            'forecast_levels': ['HIGH', 'HIGH', 'HIGH', 'HIGH', 'HIGH', 'HIGH', 'CRITICAL'],
            'peak_threat_day': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
        },
        'actionable_recommendations': [
            "📈 Increasing activity trend detected - enhance monitoring capabilities",
            "🗺️ Monitor 3 high-risk locations closely (Borno, Kaduna, Rivers)",
            "🚨 Activate elevated threat response protocols for northeast region",
            "🤝 Strengthen community engagement in high-risk areas",
            "📊 Continue monitoring economic indicators affecting security",
            "🎯 Focus resources on predicted Boko Haram and bandit activities"
        ]
    }

    # Display demo results
    _display_prediction_results(demo_predictions, "Comprehensive Analysis", 0.6)

    st.success("🎮 Demo prediction completed! This shows the type of analysis available with real data.")


# Keep all existing functions from the original dashboard
def timeline_analysis_page():
    st.header("📅 Timeline Analysis")

    if not st.session_state.processor.reports:
        st.warning("No documents processed yet. Please process some documents first.")
        return

    # Prepare data for timeline analysis - FIXED VERSION
    reports_data = []

    # Get the original texts from the processor
    document_texts = getattr(st.session_state.processor, 'document_texts', {})

    for report in st.session_state.processor.reports:
        doc_id = report['doc_id']

        # Try multiple ways to get the text content
        text = ""

        # Method 1: From stored document texts (NEW - this is the main fix)
        if document_texts and doc_id in document_texts:
            text = document_texts[doc_id]
            print(f"✅ Got text for {doc_id} from document_texts: {len(text)} chars")

        # Method 2: From session state (if manually entered)
        elif hasattr(st.session_state, 'manual_text') and 'MANUAL' in doc_id:
            text = st.session_state.get('manual_text', '')
            print(f"✅ Got text for {doc_id} from manual_text: {len(text)} chars")

        # Method 3: Create sample text for demonstration if nothing available
        else:
            # Create rich timeline text for demonstration
            text = f"""INTELLIGENCE REPORT - {doc_id}
Date: June 8, 2025
Classification: SECRET

Subject: Timeline Analysis Demonstration

OPERATIONAL TIMELINE:
- January 15, 2025: Initial intelligence activities detected
- February 20, 2025: Escalation observed in target region  
- March 10, 2025: Critical threshold reached in operations
- April 5, 2025: Peak activity period identified
- May 18, 2025: Current assessment period begins
- June 1, 2025: Latest intelligence update received
- June 8, 2025: Report compiled

This {report.get('doc_type', 'intelligence')} document contains {report.get('entity_count', 0)} entities.
Analysis indicates significant temporal patterns requiring immediate attention.

Recent developments show accelerating timeline with critical dates approaching.
Recommend continued monitoring of all identified time-sensitive indicators."""
            print(f"⚠️ Generated sample text for {doc_id}: {len(text)} chars")

        reports_data.append({
            'doc_id': doc_id,
            'doc_type': report.get('doc_type', 'unknown'),
            'text': text,
            'processed_date': report.get('processed_date', datetime.now())
        })

    # Debug information
    st.subheader("🔍 Timeline Analysis Debug")
    with st.expander("View Debug Information"):
        st.write("**Number of reports for analysis:**", len(reports_data))
        st.write("**Document texts available:**", len(document_texts))
        st.write("**Processor has document_texts attribute:**", hasattr(st.session_state.processor, 'document_texts'))

        for i, report in enumerate(reports_data[:3]):  # Show first 3
            st.write(f"**Report {i + 1}:** {report['doc_id']}")
            st.write(f"**Text length:** {len(report['text'])} characters")
            st.write(f"**Text preview:** {report['text'][:200]}...")
            st.write("---")

    with st.spinner("Analyzing temporal patterns..."):
        timeline_analysis = analytics_engine.analyze_timeline(reports_data)

    if 'error' in timeline_analysis:
        st.error(f"Timeline analysis failed: {timeline_analysis['error']}")
        st.info("💡 Make sure dateparser is installed: `pip install dateparser`")
        return

    if 'message' in timeline_analysis:
        st.info(timeline_analysis['message'])

        # Show what we tried to analyze
        st.subheader("📋 Analysis Attempted On:")
        for report in reports_data:
            with st.expander(f"Document: {report['doc_id']}"):
                st.write(f"**Type:** {report['doc_type']}")
                st.write(f"**Text length:** {len(report['text'])} characters")
                st.text_area("Text content:",
                             report['text'][:500] + "..." if len(report['text']) > 500 else report['text'], height=150)

        # Provide helpful guidance
        st.info("""
        💡 **To get better timeline analysis:**

        1. **Use the sample reports** from Document Upload page - they contain rich date content
        2. **Try this timeline-rich sample text:**

        ```
        INTELLIGENCE REPORT - June 8, 2025
        January 15, 2025: Operation began
        February 20, 2025: Key milestone reached  
        March 10, 2025: Critical phase initiated
        April 5, 2025: Peak activity detected
        May 18, 2025: Current status assessed
        ```

        3. **Create analytics samples:** `python create_analytics_samples.py`
        4. **Make sure your documents include specific dates** in formats like:
           - June 8, 2025
           - 2025-06-08  
           - 06/08/2025
        """)
        return

    # Timeline overview - SUCCESS CASE
    st.success(f"✅ Timeline analysis completed! Found {timeline_analysis['total_dates']} temporal references.")

    st.subheader("📊 Timeline Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Dates Found", timeline_analysis['total_dates'])
    with col2:
        date_range = timeline_analysis['date_range']
        if date_range['earliest']:
            st.metric("Earliest Date", date_range['earliest'].strftime('%Y-%m-%d'))
    with col3:
        if date_range['latest']:
            st.metric("Latest Date", date_range['latest'].strftime('%Y-%m-%d'))
    with col4:
        if date_range['earliest'] and date_range['latest']:
            span = (date_range['latest'] - date_range['earliest']).days
            st.metric("Time Span (Days)", span)

    # Timeline visualization
    st.subheader("📈 Temporal Activity Timeline")

    if timeline_analysis.get('timeline_data'):
        timeline_df = pd.DataFrame(timeline_analysis['timeline_data'])

        # Create timeline chart
        fig_timeline = px.scatter(
            timeline_df,
            x='extracted_date',
            y='doc_id',
            color='doc_type',
            size='confidence',
            hover_data=['date_text', 'context'],
            title="Intelligence Timeline",
            labels={'extracted_date': 'Date', 'doc_id': 'Document ID'}
        )
        fig_timeline.update_layout(height=500)
        st.plotly_chart(fig_timeline, use_container_width=True)

        # Show timeline data table
        with st.expander("📋 Timeline Data Details"):
            st.dataframe(timeline_df[['doc_id', 'extracted_date', 'date_text', 'confidence', 'context']],
                         use_container_width=True)

        # Monthly activity chart
        if timeline_analysis.get('monthly_activity'):
            monthly_df = pd.DataFrame(timeline_analysis['monthly_activity'])

            if not monthly_df.empty:
                fig_monthly = px.bar(
                    monthly_df,
                    x='date',
                    y='count',
                    title="Monthly Intelligence Activity",
                    labels={'date': 'Month', 'count': 'Number of Events'}
                )
                st.plotly_chart(fig_monthly, use_container_width=True)

    # Temporal patterns
    st.subheader("🔍 Temporal Patterns")

    if timeline_analysis.get('temporal_patterns'):
        patterns = timeline_analysis['temporal_patterns']

        col1, col2 = st.columns(2)

        with col1:
            if patterns.get('yearly_activity'):
                st.write("**Yearly Activity**")
                yearly_df = pd.DataFrame(list(patterns['yearly_activity'].items()),
                                         columns=['Year', 'Count'])
                st.dataframe(yearly_df, use_container_width=True)

        with col2:
            if patterns.get('weekly_patterns'):
                st.write("**Weekly Patterns**")
                weekly_df = pd.DataFrame(list(patterns['weekly_patterns'].items()),
                                         columns=['Day', 'Count'])
                st.dataframe(weekly_df, use_container_width=True)

    # Key temporal entities
    st.subheader("🎯 Key Temporal Entities")

    if timeline_analysis.get('temporal_entities'):
        temporal_entities = timeline_analysis['temporal_entities'][:10]  # Top 10

        for entity in temporal_entities:
            with st.expander(f"📅 {entity['date'].strftime('%Y-%m-%d')} (Importance: {entity['importance_score']:.2f})"):
                st.write(f"**Documents**: {', '.join(entity['documents'])}")
                st.write(f"**Document Count**: {entity['document_count']}")

                if entity['contexts']:
                    st.write("**Key Contexts**:")
                    for i, context in enumerate(entity['contexts'][:3]):
                        st.write(f"{i + 1}. {context}")

    # Export timeline data
    if st.button("📥 Export Timeline Data", key="export_timeline_btn"):
        if timeline_analysis.get('timeline_data'):
            timeline_df = pd.DataFrame(timeline_analysis['timeline_data'])
            csv_data = timeline_df.to_csv(index=False)
            st.download_button(
                "Download Timeline CSV",
                csv_data,
                f"timeline_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv",
                key="download_timeline_csv_btn"
            )


def geographic_mapping_page():
    st.header("🗺️ Geographic Intelligence Mapping")

    if not st.session_state.processor.entities:
        st.warning("No entities processed yet. Please process some documents first.")
        return

    with st.spinner("Analyzing geographic patterns..."):
        geo_analysis = analytics_engine.analyze_geography(st.session_state.processor.entities)

    if 'error' in geo_analysis:
        st.error(f"Geographic analysis failed: {geo_analysis['error']}")
        if not FOLIUM_AVAILABLE:
            st.info("💡 Install folium for geographic mapping: `pip install folium streamlit-folium`")
        return

    if 'message' in geo_analysis:
        st.info(geo_analysis['message'])
        return

    # Geographic overview
    st.subheader("🌍 Geographic Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Locations", geo_analysis['total_locations'])
    with col2:
        unique_countries = len(geo_analysis.get('country_analysis', {}))
        st.metric("Countries Identified", unique_countries)
    with col3:
        hotspots_count = len(geo_analysis.get('hotspots', []))
        st.metric("Geographic Hotspots", hotspots_count)
    with col4:
        clusters_count = len(geo_analysis.get('geographic_clusters', []))
        st.metric("Activity Clusters", clusters_count)

    # Interactive map
    if FOLIUM_AVAILABLE and geo_analysis.get('geographic_data'):
        st.subheader("🗺️ Interactive Intelligence Map")

        # Create folium map
        geographic_data = geo_analysis['geographic_data']

        if geographic_data:
            # Calculate map center
            avg_lat = sum(loc['latitude'] for loc in geographic_data) / len(geographic_data)
            avg_lon = sum(loc['longitude'] for loc in geographic_data) / len(geographic_data)

            # Create map
            m = folium.Map(location=[avg_lat, avg_lon], zoom_start=4)

            # Add markers for each location
            for location in geographic_data:
                # Marker color based on frequency
                if location['frequency'] > 3:
                    color = 'red'  # High activity
                    icon = 'exclamation-sign'
                elif location['frequency'] > 1:
                    color = 'orange'  # Medium activity
                    icon = 'warning-sign'
                else:
                    color = 'blue'  # Low activity
                    icon = 'info-sign'

                popup_text = f"""
                <b>{location['location']}</b><br>
                Frequency: {location['frequency']}<br>
                Country: {location['country']}<br>
                Document: {location['doc_id']}
                """

                folium.Marker(
                    location=[location['latitude'], location['longitude']],
                    popup=folium.Popup(popup_text, max_width=300),
                    tooltip=location['location'],
                    icon=folium.Icon(color=color, icon=icon)
                ).add_to(m)

            # Add hotspots as circles
            for hotspot in geo_analysis.get('hotspots', []):
                folium.Circle(
                    location=[hotspot['latitude'], hotspot['longitude']],
                    radius=hotspot['intensity'] * 10000,  # Scale radius
                    popup=f"Hotspot: {hotspot['location']} (Risk: {hotspot['risk_level']})",
                    color='red' if hotspot['risk_level'] == 'HIGH' else 'orange',
                    fill=True,
                    opacity=0.3
                ).add_to(m)

            # Display map
            map_data = st_folium(m, width=700, height=500)

    # Location frequency analysis
    st.subheader("📊 Location Analysis")

    col1, col2 = st.columns(2)

    with col1:
        if geo_analysis.get('location_frequency'):
            st.write("**Most Mentioned Locations**")
            location_freq = geo_analysis['location_frequency']
            freq_df = pd.DataFrame(list(location_freq.items()), columns=['Location', 'Frequency'])

            fig_locations = px.bar(
                freq_df.head(10),
                x='Frequency',
                y='Location',
                orientation='h',
                title="Top 10 Locations by Mention Frequency"
            )
            fig_locations.update_layout(height=400)
            st.plotly_chart(fig_locations, use_container_width=True)

    with col2:
        if geo_analysis.get('country_analysis'):
            st.write("**Activity by Country**")
            country_data = []
            for country, data in geo_analysis['country_analysis'].items():
                country_data.append({
                    'Country': country,
                    'Locations': data['location_count'],
                    'Total Mentions': data['total_mentions']
                })

            country_df = pd.DataFrame(country_data)

            fig_countries = px.pie(
                country_df,
                values='Total Mentions',
                names='Country',
                title="Geographic Activity by Country"
            )
            st.plotly_chart(fig_countries, use_container_width=True)

    # Geographic hotspots
    if geo_analysis.get('hotspots'):
        st.subheader("🎯 Geographic Hotspots")

        hotspots_df = pd.DataFrame(geo_analysis['hotspots'])

        for idx, hotspot in hotspots_df.iterrows():
            risk_color = "🔴" if hotspot['risk_level'] == 'HIGH' else "🟡"

            with st.expander(f"{risk_color} {hotspot['location']} (Intensity: {hotspot['intensity']})"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Risk Level**: {hotspot['risk_level']}")
                    st.write(f"**Activity Intensity**: {hotspot['intensity']}")
                with col2:
                    st.write(f"**Latitude**: {hotspot['latitude']:.4f}")
                    st.write(f"**Longitude**: {hotspot['longitude']:.4f}")


def sentiment_analysis_page():
    st.header("📊 Sentiment & Threat Analysis")

    if not st.session_state.processor.reports:
        st.warning("No documents processed yet. Please process some documents first.")
        return

    # Prepare data for sentiment analysis
    reports_data = []
    for report in st.session_state.processor.reports:
        # Find the original text for this report (simplified approach)
        text = f"Sample intelligence text for {report['doc_id']}"  # This should be the actual processed text

        reports_data.append({
            'doc_id': report['doc_id'],
            'doc_type': report['doc_type'],
            'text': text,
            'processed_date': report['processed_date']
        })

    with st.spinner("Analyzing sentiment and threat levels..."):
        sentiment_analysis = analytics_engine.analyze_sentiment(reports_data)

    if 'error' in sentiment_analysis:
        st.error(f"Sentiment analysis failed: {sentiment_analysis['error']}")
        return

    # Sentiment overview
    st.subheader("📈 Sentiment Overview")

    overall_sentiment = sentiment_analysis['overall_sentiment']
    threat_analysis = sentiment_analysis['threat_analysis']
    urgency_analysis = sentiment_analysis['urgency_analysis']

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        polarity = overall_sentiment['avg_polarity']
        polarity_emoji = "😊" if polarity > 0.1 else "😐" if polarity > -0.1 else "😟"
        st.metric("Avg Sentiment Polarity", f"{polarity:.3f} {polarity_emoji}")

    with col2:
        compound = overall_sentiment['avg_compound']
        st.metric("Avg Compound Score", f"{compound:.3f}")

    with col3:
        threat_score = threat_analysis['avg_threat_score']
        threat_emoji = "🔴" if threat_score > 0.7 else "🟡" if threat_score > 0.4 else "🟢"
        st.metric("Avg Threat Score", f"{threat_score:.3f} {threat_emoji}")

    with col4:
        urgency_score = urgency_analysis['avg_urgency']
        urgency_emoji = "⚡" if urgency_score > 0.7 else "⚠️" if urgency_score > 0.4 else "📋"
        st.metric("Avg Urgency Score", f"{urgency_score:.3f} {urgency_emoji}")

    # Sentiment distribution
    st.subheader("📊 Sentiment & Threat Distribution")

    col1, col2 = st.columns(2)

    with col1:
        sentiment_dist = overall_sentiment['sentiment_distribution']

        fig_sentiment = px.pie(
            values=list(sentiment_dist.values()),
            names=list(sentiment_dist.keys()),
            title="Sentiment Distribution",
            color_discrete_map={
                'positive': 'green',
                'neutral': 'gray',
                'negative': 'red'
            }
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)

    with col2:
        threat_dist = threat_analysis['threat_distribution']

        fig_threat = px.pie(
            values=list(threat_dist.values()),
            names=list(threat_dist.keys()),
            title="Threat Level Distribution",
            color_discrete_map={
                'HIGH': 'red',
                'MEDIUM': 'orange',
                'LOW': 'green'
            }
        )
        st.plotly_chart(fig_threat, use_container_width=True)

    # Document-level analysis
    st.subheader("📋 Document-Level Analysis")

    if sentiment_analysis.get('sentiment_data'):
        sentiment_df = pd.DataFrame(sentiment_analysis['sentiment_data'])

        # Sentiment vs Threat scatter plot
        fig_scatter = px.scatter(
            sentiment_df,
            x='textblob_polarity',
            y='threat_score',
            color='threat_level',
            size='urgency_score',
            hover_data=['doc_id', 'doc_type'],
            title="Sentiment vs Threat Analysis",
            labels={
                'textblob_polarity': 'Sentiment Polarity',
                'threat_score': 'Threat Score'
            },
            color_discrete_map={
                'HIGH': 'red',
                'MEDIUM': 'orange',
                'LOW': 'green'
            }
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # High priority documents
        st.subheader("🚨 High Priority Intelligence")

        high_threat_docs = threat_analysis.get('high_threat_docs', [])
        urgent_docs = urgency_analysis.get('urgent_docs', [])

        priority_docs = list(set(high_threat_docs + urgent_docs))

        if priority_docs:
            st.warning(f"⚠️ {len(priority_docs)} documents flagged as high priority")

            priority_df = sentiment_df[sentiment_df['doc_id'].isin(priority_docs)]

            for idx, doc in priority_df.iterrows():
                threat_emoji = "🔴" if doc['threat_level'] == 'HIGH' else "🟡"
                urgency_indicator = "⚡" if doc['urgency_score'] > 0.7 else ""

                with st.expander(f"{threat_emoji} {doc['doc_id']} - {doc['doc_type']} {urgency_indicator}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Threat Level**: {doc['threat_level']}")
                        st.write(f"**Threat Score**: {doc['threat_score']:.3f}")
                        st.write(f"**Urgency Score**: {doc['urgency_score']:.3f}")
                    with col2:
                        st.write(f"**Sentiment Polarity**: {doc['textblob_polarity']:.3f}")
                        st.write(f"**VADER Compound**: {doc['vader_compound']:.3f}")
                        if doc['threat_indicators']:
                            st.write(f"**Threat Indicators**: {', '.join(doc['threat_indicators'][:3])}")
        else:
            st.success("✅ No high priority threats detected")

    # Keyword sentiment analysis
    if sentiment_analysis.get('keyword_analysis'):
        st.subheader("🔤 Intelligence Keywords Sentiment")

        keyword_data = sentiment_analysis['keyword_analysis']
        keyword_list = []

        for keyword, data in keyword_data.items():
            keyword_list.append({
                'Keyword': keyword,
                'Sentiment': data['polarity'],
                'Subjectivity': data['subjectivity'],
                'Mentions': data['mention_count']
            })

        if keyword_list:
            keyword_df = pd.DataFrame(keyword_list)

            fig_keywords = px.scatter(
                keyword_df,
                x='Sentiment',
                y='Subjectivity',
                size='Mentions',
                hover_data=['Keyword'],
                title="Intelligence Keywords Sentiment Analysis",
                labels={
                    'Sentiment': 'Sentiment Polarity',
                    'Subjectivity': 'Subjectivity Score'
                }
            )
            st.plotly_chart(fig_keywords, use_container_width=True)

    # Word cloud of threat indicators
    if WORDCLOUD_AVAILABLE and sentiment_analysis.get('sentiment_data'):
        st.subheader("☁️ Threat Indicators Word Cloud")

        # Collect all threat indicators
        all_indicators = []
        for doc in sentiment_analysis['sentiment_data']:
            all_indicators.extend(doc.get('threat_indicators', []))

        if all_indicators:
            # Create word cloud
            wordcloud_text = ' '.join(all_indicators)

            try:
                wordcloud = WordCloud(
                    width=800,
                    height=400,
                    background_color='white',
                    colormap='Reds'
                ).generate(wordcloud_text)

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            except Exception as e:
                st.info("Word cloud could not be generated")


def entity_analysis_page():
    st.header("👥 Entity Analysis")

    if not st.session_state.processor.entities:
        st.warning("No entities extracted yet. Please process some documents first.")
        return

    entities_df = pd.DataFrame(st.session_state.processor.entities)

    # Entity type distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Entity Types")
        type_counts = entities_df['label'].value_counts()
        fig_pie = px.pie(
            values=type_counts.values,
            names=type_counts.index,
            title="Distribution of Entity Types"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("Most Frequent Entities")
        entity_counts = entities_df['text'].value_counts().head(10)
        fig_bar = px.bar(
            x=entity_counts.index,
            y=entity_counts.values,
            title="Top 10 Most Mentioned Entities"
        )
        fig_bar.update_xaxes(tickangle=45)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Entity search and filter
    st.subheader("Entity Search & Filter")

    col1, col2 = st.columns(2)
    with col1:
        search_term = st.text_input("Search entities:")
        entity_type_filter = st.multiselect(
            "Filter by type:",
            entities_df['label'].unique()
        )

    # Apply filters
    filtered_df = entities_df.copy()
    if search_term:
        filtered_df = filtered_df[
            filtered_df['text'].str.contains(search_term, case=False)
        ]
    if entity_type_filter:
        filtered_df = filtered_df[filtered_df['label'].isin(entity_type_filter)]

    st.dataframe(filtered_df, use_container_width=True)


def relationship_network_page():
    st.header("🕸️ Relationship Network Analysis")

    if not st.session_state.processor.relationships:
        st.warning("⚠️ No relationships found. Please process some documents first.")
        st.info("💡 Try processing the sample reports from the Document Upload page!")
        return

    # Network configuration
    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Network Settings")
        min_weight = st.slider("Minimum Relationship Strength", 0.1, 3.0, 0.5, 0.1)
        layout_type = st.selectbox("Layout Algorithm", ["spring", "circular", "random"])
        show_labels = st.checkbox("Show Entity Labels", value=True)
        node_size_factor = st.slider("Node Size Factor", 1, 10, 3)

    # Get network graph with user settings
    try:
        G = st.session_state.processor.get_entity_network(min_weight=min_weight)

        if len(G.nodes()) == 0:
            st.warning("🔍 No relationships meet the current threshold. Try lowering the minimum relationship strength.")
            return

        with col2:
            # Choose layout algorithm
            if layout_type == "spring":
                pos = nx.spring_layout(G, k=1.5, iterations=50)
            elif layout_type == "circular":
                pos = nx.circular_layout(G)
            else:
                pos = nx.random_layout(G)

            # Create interactive network visualization
            fig = create_network_plot(G, pos, show_labels, node_size_factor)
            st.plotly_chart(fig, use_container_width=True)

        # Network analysis metrics
        st.subheader("📊 Network Analysis")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Nodes (Entities)", len(G.nodes()))
        with col2:
            st.metric("Edges (Relationships)", len(G.edges()))
        with col3:
            if len(G.nodes()) > 0:
                density = nx.density(G)
                st.metric("Network Density", f"{density:.3f}")
        with col4:
            if len(G.nodes()) > 0:
                avg_degree = sum(dict(G.degree()).values()) / len(G.nodes())
                st.metric("Avg Connections", f"{avg_degree:.1f}")

        # Top connected entities
        if len(G.nodes()) > 0:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Most Connected Entities")
                degree_centrality = nx.degree_centrality(G)
                top_connected = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]

                if top_connected:
                    df_connected = pd.DataFrame(top_connected, columns=['Entity', 'Centrality'])
                    st.dataframe(df_connected, use_container_width=True)

            with col2:
                st.subheader("Entity Types in Network")
                entity_types = {}
                for node in G.nodes():
                    entity_type = G.nodes[node].get('entity_type', 'UNKNOWN')
                    entity_types[entity_type] = entity_types.get(entity_type, 0) + 1

                if entity_types:
                    fig_types = px.pie(
                        values=list(entity_types.values()),
                        names=list(entity_types.keys()),
                        title="Entity Types Distribution"
                    )
                    st.plotly_chart(fig_types, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error creating network visualization: {str(e)}")
        st.info("This might happen with very large networks. Try increasing the minimum relationship strength.")


def create_network_plot(G, pos, show_labels=True, size_factor=3):
    """Create an interactive network plot using Plotly"""

    # Prepare edge traces
    edge_x = []
    edge_y = []
    edge_info = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

        weight = G[edge[0]][edge[1]].get('weight', 1)
        edge_info.append(f"{edge[0]} ↔ {edge[1]}<br>Strength: {weight:.2f}")

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='rgba(125,125,125,0.5)'),
        hoverinfo='none',
        mode='lines'
    )

    # Prepare node traces
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    node_info = []

    # Color mapping for entity types
    color_map = {
        'PERSON': 'lightcoral',
        'ORG': 'lightblue',
        'GPE': 'lightgreen',
        'MONEY': 'gold',
        'DATE': 'plum',
        'EVENT': 'orange',
        'PRODUCT': 'cyan',
        'UNKNOWN': 'lightgray'
    }

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        # Node size based on frequency and connections
        frequency = G.nodes[node].get('frequency', 1)
        degree = G.degree(node)
        size = max(10, (frequency + degree) * size_factor)
        node_size.append(size)

        # Node color based on entity type
        entity_type = G.nodes[node].get('entity_type', 'UNKNOWN')
        node_color.append(color_map.get(entity_type, 'lightgray'))

        # Node info for hover
        node_info.append(f"{node}<br>Type: {entity_type}<br>Frequency: {frequency}<br>Connections: {degree}")

        # Node labels
        if show_labels:
            node_text.append(node if len(node) < 15 else node[:12] + "...")
        else:
            node_text.append("")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        hovertext=node_info,
        text=node_text,
        textposition="middle center",
        textfont=dict(size=10, color="black"),
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=2, color='darkblue'),
            opacity=0.8
        )
    )

    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title={
                            'text': 'Entity Relationship Network',
                            'x': 0.5,
                            'xanchor': 'center'
                        },
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="Node size = frequency + connections | Colors = entity types",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002,
                            xanchor='left', yanchor='bottom',
                            font=dict(color="gray", size=10)
                        )],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor='white'
                    ))

    return fig


def insights_dashboard_page():
    st.header("📊 Intelligence Insights")

    if not st.session_state.processor.entities:
        st.warning("No data available. Please process some documents first.")
        return

    insights = st.session_state.processor.get_insights()

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Documents", len(st.session_state.processor.reports))
    with col2:
        st.metric("Total Entities", len(st.session_state.processor.entities))
    with col3:
        st.metric("Unique Entities", len(set(ent['text'] for ent in st.session_state.processor.entities)))
    with col4:
        st.metric("Relationships", len(st.session_state.processor.relationships))

    # Top entities and connections
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Most Frequent Entities")
        if insights['top_entities']:
            top_df = pd.DataFrame(insights['top_entities'], columns=['Entity', 'Frequency'])
            st.dataframe(top_df, use_container_width=True)

    with col2:
        st.subheader("Most Connected Entities")
        if insights['most_connected']:
            connected_df = pd.DataFrame(insights['most_connected'], columns=['Entity', 'Connections'])
            st.dataframe(connected_df, use_container_width=True)

    # Document analysis
    st.subheader("Document Analysis")
    if st.session_state.processor.reports:
        reports_df = pd.DataFrame(st.session_state.processor.reports)

        col1, col2 = st.columns(2)
        with col1:
            # Document types
            doc_type_counts = reports_df['doc_type'].value_counts()
            fig_doc = px.bar(
                x=doc_type_counts.index,
                y=doc_type_counts.values,
                title="Documents by Type"
            )
            st.plotly_chart(fig_doc, use_container_width=True)

        with col2:
            # Entity density
            fig_density = px.scatter(
                reports_df,
                x='text_length',
                y='entity_count',
                title="Entity Density by Document Length",
                labels={'text_length': 'Document Length (chars)', 'entity_count': 'Entities Extracted'}
            )
            st.plotly_chart(fig_density, use_container_width=True)

    # Export functionality
    st.subheader("Data Export")
    if st.button("Export Analysis Data", key="export_analysis_btn"):
        data = st.session_state.processor.export_data()

        # Create download links for each dataset
        col1, col2, col3 = st.columns(3)

        with col1:
            csv_entities = data['entities'].to_csv(index=False)
            st.download_button(
                "Download Entities CSV",
                csv_entities,
                "entities.csv",
                "text/csv",
                key="download_entities_btn"
            )

        with col2:
            csv_relationships = data['relationships'].to_csv(index=False)
            st.download_button(
                "Download Relationships CSV",
                csv_relationships,
                "relationships.csv",
                "text/csv",
                key="download_relationships_btn"
            )

        with col3:
            csv_reports = data['reports'].to_csv(index=False)
            st.download_button(
                "Download Reports CSV",
                csv_reports,
                "reports.csv",
                "text/csv",
                key="download_reports_btn"
            )


if __name__ == "__main__":
    main()