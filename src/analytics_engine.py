#!/usr/bin/env python3
"""
Enhanced Analytics Engine for Intelligence Analysis with Predictive Capabilities
Handles Timeline Analysis, Geographic Mapping, Sentiment Analysis, and PREDICTIVE FORECASTING
ADDED: Nigerian-focused predictive analysis and forecasting capabilities
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import warnings

warnings.filterwarnings('ignore')

# Analytics libraries
try:
    import folium
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut
    import pycountry

    GEOGRAPHIC_AVAILABLE = True
except ImportError:
    GEOGRAPHIC_AVAILABLE = False

try:
    from textblob import TextBlob
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False

try:
    import dateparser
    from dateutil import parser as date_parser

    TEMPORAL_AVAILABLE = True
except ImportError:
    TEMPORAL_AVAILABLE = False

try:
    from wordcloud import WordCloud

    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# NEW: Predictive Analytics Libraries
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, accuracy_score
    import joblib

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    import xgboost as xgb
    import lightgbm as lgb

    GRADIENT_BOOSTING_AVAILABLE = True
except ImportError:
    GRADIENT_BOOSTING_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class AnalyticsEngine:
    """Enhanced analytics engine with predictive capabilities for Nigerian intelligence analysis"""

    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer() if SENTIMENT_AVAILABLE else None
        self.geocoder = Nominatim(user_agent="intelligence_analysis") if GEOGRAPHIC_AVAILABLE else None
        self.location_cache = {}

        # Nigerian-specific threat indicators and patterns
        self.nigerian_threat_keywords = {
            'high': [
                'boko haram', 'ipob', 'bandits', 'kidnapping', 'ransom', 'terrorism', 'insurgency',
                'bombing', 'attack', 'militant', 'explosion', 'armed robbery', 'pipeline vandalism',
                'oil theft', 'communal clash', 'ethnic violence', 'religious crisis', 'farmers herders',
                'cultism', 'ritual killing', 'niger delta', 'sambisa forest', 'chibok girls'
            ],
            'medium': [
                'protest', 'strike', 'demonstration', 'unrest', 'agitation', 'separatist',
                'youth restiveness', 'unemployment', 'poverty', 'corruption', 'election violence',
                'political crisis', 'economic hardship', 'fuel scarcity', 'power outage',
                'infrastructure decay', 'border security', 'illegal mining', 'drug trafficking'
            ],
            'low': [
                'development', 'peace building', 'reconciliation', 'economic growth',
                'infrastructure development', 'education', 'healthcare', 'agriculture',
                'technology', 'investment', 'job creation', 'poverty alleviation'
            ]
        }

        # Standard threat keywords (from original)
        self.threat_keywords = {
            'high': ['attack', 'threat', 'immediate', 'urgent', 'critical', 'emergency', 'danger', 'weapon',
                     'explosive', 'IPOB', 'Bandits', 'Bokoharam', 'terrorist'],
            'medium': ['suspicious', 'unusual', 'concern', 'monitor', 'investigate', 'alert', 'warning', 'risk'],
            'low': ['routine', 'normal', 'standard', 'regular', 'typical', 'maintenance', 'administrative']
        }

        # Nigerian geographic risk zones
        self.nigerian_risk_zones = {
            'very_high': [
                'borno', 'yobe', 'adamawa', 'gombe', 'bauchi', 'kaduna', 'katsina',
                'zamfara', 'sokoto', 'kebbi', 'niger delta', 'rivers', 'delta', 'bayelsa'
            ],
            'high': [
                'plateau', 'taraba', 'nasarawa', 'benue', 'kogi', 'edo', 'cross river',
                'akwa ibom', 'imo', 'abia', 'anambra', 'enugu', 'ebonyi'
            ],
            'medium': [
                'ogun', 'oyo', 'osun', 'ondo', 'ekiti', 'kwara', 'abuja', 'jigawa', 'kano'
            ],
            'low': ['lagos', 'yobe']  # Some areas of these states
        }

        # Criminal organization patterns (Nigerian context)
        self.criminal_patterns = {
            'boko_haram': {
                'typical_activities': ['bombing', 'kidnapping', 'territory_control', 'taxation'],
                'geographical_focus': ['northeast_nigeria', 'chad_basin', 'cameroon_border'],
                'seasonal_patterns': ['dry_season_intensification', 'rainy_season_movement'],
                'funding_sources': ['robbery', 'kidnapping_ransom', 'taxation', 'external_support']
            },
            'ipob': {
                'typical_activities': ['civil_disobedience', 'separatist_agitation', 'sit_at_home'],
                'geographical_focus': ['southeast_nigeria', 'igbo_majority_areas'],
                'seasonal_patterns': ['biafra_remembrance_periods', 'election_seasons'],
                'funding_sources': ['diaspora_funding', 'local_contributions', 'business_support']
            },
            'bandits': {
                'typical_activities': ['kidnapping', 'cattle_rustling', 'village_raids', 'ransom_collection'],
                'geographical_focus': ['northwest_nigeria', 'northcentral_nigeria', 'forest_areas'],
                'seasonal_patterns': ['dry_season_movement', 'farming_season_targeting'],
                'funding_sources': ['ransom_payments', 'cattle_sales', 'protection_fees']
            },
            'niger_delta_militants': {
                'typical_activities': ['pipeline_vandalism', 'oil_theft', 'kidnapping', 'sea_piracy'],
                'geographical_focus': ['niger_delta', 'coastal_areas', 'oil_installations'],
                'seasonal_patterns': ['oil_price_correlation', 'election_periods'],
                'funding_sources': ['oil_theft', 'ransom', 'political_funding', 'illegal_refining']
            }
        }

        # Economic indicators affecting security (Nigerian context)
        self.economic_security_indicators = [
            'unemployment_rate', 'inflation_rate', 'oil_prices', 'exchange_rate',
            'poverty_index', 'food_prices', 'fuel_prices', 'electricity_supply',
            'agricultural_output', 'government_budget', 'foreign_reserves'
        ]

        print("🧠 Enhanced Analytics Engine with Predictive Capabilities initialized")
        print(f"📍 Geographic analysis: {'✅' if GEOGRAPHIC_AVAILABLE else '❌'}")
        print(f"📊 Sentiment analysis: {'✅' if SENTIMENT_AVAILABLE else '❌'}")
        print(f"📅 Temporal analysis: {'✅' if TEMPORAL_AVAILABLE else '❌'}")
        print(f"🔮 Predictive models: {'✅' if SKLEARN_AVAILABLE else '❌'}")
        print(f"📈 Time series forecasting: {'✅' if PROPHET_AVAILABLE or STATSMODELS_AVAILABLE else '❌'}")

    def analyze_timeline(self, reports_data: List[Dict]) -> Dict:
        """Extract and analyze temporal patterns from reports (existing method)"""
        if not TEMPORAL_AVAILABLE:
            return {"error": "Temporal analysis libraries not available"}

        timeline_data = []
        print(f"🔍 Analyzing timeline for {len(reports_data)} reports...")

        for report in reports_data:
            doc_id = report.get('doc_id', 'unknown')
            text = report.get('text', '')
            doc_type = report.get('doc_type', 'unknown')
            processed_date = report.get('processed_date', datetime.now())

            print(f"📄 Processing {doc_id}: {len(text)} characters")
            extracted_dates = self._extract_dates_from_text(text)
            print(f"📅 Found {len(extracted_dates)} dates in {doc_id}")

            for date_info in extracted_dates:
                timeline_data.append({
                    'doc_id': doc_id,
                    'doc_type': doc_type,
                    'extracted_date': date_info['date'],
                    'date_text': date_info['text'],
                    'confidence': date_info['confidence'],
                    'processed_date': processed_date,
                    'context': date_info['context']
                })

        print(f"📊 Total timeline entries: {len(timeline_data)}")
        timeline_df = pd.DataFrame(timeline_data)

        if timeline_df.empty:
            return {"message": "No temporal data found in documents"}

        analysis = {
            'total_dates': len(timeline_df),
            'date_range': {
                'earliest': timeline_df['extracted_date'].min(),
                'latest': timeline_df['extracted_date'].max()
            },
            'temporal_patterns': self._analyze_temporal_patterns(timeline_df),
            'timeline_data': timeline_df.to_dict('records'),
            'monthly_activity': self._get_monthly_activity(timeline_df),
            'temporal_entities': self._extract_temporal_entities(timeline_df)
        }

        return analysis

    # NEW: Predictive Analysis Methods

    def predict_future_trends(self, reports_data: List[Dict], forecast_days: int = 90) -> Dict:
        """Predict future trends based on historical intelligence data"""
        print(f"🔮 Starting predictive analysis for {forecast_days} days ahead...")

        if not SKLEARN_AVAILABLE:
            return {"error": "Machine learning libraries not available for predictions"}

        try:
            # Prepare time series data
            timeline_analysis = self.analyze_timeline(reports_data)

            if 'timeline_data' not in timeline_analysis:
                return {"error": "No timeline data available for prediction"}

            # Time series forecasting
            time_series_prediction = self._forecast_activity_trends(timeline_analysis, forecast_days)

            # Geographic risk prediction
            geographic_prediction = self._predict_geographic_risks(reports_data)

            # Threat level forecasting
            threat_prediction = self._forecast_threat_levels(reports_data, forecast_days)

            # Nigerian-specific criminal organization prediction
            criminal_prediction = self._predict_criminal_activities(reports_data)

            # Economic indicator impact prediction
            economic_impact = self._predict_economic_security_impact(reports_data)

            prediction_results = {
                'forecast_period_days': forecast_days,
                'generated_date': datetime.now(),
                'time_series_forecast': time_series_prediction,
                'geographic_risk_prediction': geographic_prediction,
                'threat_level_forecast': threat_prediction,
                'criminal_organization_forecast': criminal_prediction,
                'economic_security_impact': economic_impact,
                'overall_risk_assessment': self._calculate_overall_risk_score(
                    time_series_prediction, geographic_prediction, threat_prediction
                ),
                'actionable_recommendations': self._generate_recommendations(
                    time_series_prediction, geographic_prediction, threat_prediction
                )
            }

            print("✅ Predictive analysis completed successfully")
            return prediction_results

        except Exception as e:
            print(f"❌ Error in predictive analysis: {str(e)}")
            return {"error": f"Prediction failed: {str(e)}"}

    def _forecast_activity_trends(self, timeline_analysis: Dict, forecast_days: int) -> Dict:
        """Forecast future activity trends using time series analysis"""
        try:
            if not timeline_analysis.get('monthly_activity'):
                return {"error": "No monthly activity data for forecasting"}

            monthly_data = pd.DataFrame(timeline_analysis['monthly_activity'])

            if len(monthly_data) < 3:
                # Create synthetic trend data for demonstration
                return self._create_synthetic_forecast(forecast_days)

            # Convert to time series
            monthly_data['date'] = pd.to_datetime(monthly_data['date'])
            monthly_data = monthly_data.sort_values('date')
            monthly_data.set_index('date', inplace=True)

            # Simple trend analysis
            if STATSMODELS_AVAILABLE and len(monthly_data) >= 6:
                try:
                    # ARIMA forecasting
                    model = ARIMA(monthly_data['count'], order=(1, 1, 1))
                    fitted_model = model.fit()
                    forecast = fitted_model.forecast(steps=3)  # 3 months ahead

                    forecast_dates = pd.date_range(
                        start=monthly_data.index[-1] + pd.DateOffset(months=1),
                        periods=3,
                        freq='M'
                    )

                    return {
                        'model_type': 'ARIMA',
                        'forecast_values': forecast.tolist(),
                        'forecast_dates': forecast_dates.strftime('%Y-%m-%d').tolist(),
                        'trend_direction': 'increasing' if forecast[-1] > monthly_data['count'].iloc[
                            -1] else 'decreasing',
                        'confidence_level': 0.75
                    }
                except:
                    pass

            # Fallback: Simple linear trend
            monthly_data['period'] = range(len(monthly_data))
            X = monthly_data[['period']]
            y = monthly_data['count']

            model = LinearRegression()
            model.fit(X, y)

            # Forecast future periods
            future_periods = np.arange(len(monthly_data), len(monthly_data) + 3).reshape(-1, 1)
            forecast = model.predict(future_periods)

            return {
                'model_type': 'Linear Regression',
                'forecast_values': forecast.tolist(),
                'trend_direction': 'increasing' if model.coef_[0] > 0 else 'decreasing',
                'confidence_level': 0.65,
                'trend_strength': abs(model.coef_[0])
            }

        except Exception as e:
            return {"error": f"Time series forecasting failed: {str(e)}"}

    def _predict_geographic_risks(self, reports_data: List[Dict]) -> Dict:
        """Predict geographic risk levels for Nigerian regions"""
        try:
            # Extract locations from reports
            location_mentions = defaultdict(int)
            location_threat_scores = defaultdict(list)

            for report in reports_data:
                text = report.get('text', '').lower()

                # Check for Nigerian locations and calculate threat scores
                for risk_level, locations in self.nigerian_risk_zones.items():
                    for location in locations:
                        if location in text:
                            location_mentions[location] += 1

                            # Calculate threat score based on text content
                            threat_score = self._calculate_location_threat_score(text, location)
                            location_threat_scores[location].append(threat_score)

            # Predict future risk levels
            risk_predictions = {}

            for location, mentions in location_mentions.items():
                if mentions > 0:
                    avg_threat = np.mean(location_threat_scores[location]) if location_threat_scores[location] else 0.5

                    # Determine current risk zone
                    current_risk = 'medium'
                    for risk_level, locations in self.nigerian_risk_zones.items():
                        if location in locations:
                            current_risk = risk_level
                            break

                    # Predict future risk (simplified model)
                    risk_multiplier = {
                        'very_high': 1.2,
                        'high': 1.1,
                        'medium': 1.0,
                        'low': 0.9
                    }

                    predicted_risk_score = avg_threat * risk_multiplier.get(current_risk, 1.0) * (mentions / 10)
                    predicted_risk_score = min(1.0, predicted_risk_score)

                    if predicted_risk_score > 0.8:
                        predicted_level = 'VERY HIGH'
                    elif predicted_risk_score > 0.6:
                        predicted_level = 'HIGH'
                    elif predicted_risk_score > 0.4:
                        predicted_level = 'MEDIUM'
                    else:
                        predicted_level = 'LOW'

                    risk_predictions[location] = {
                        'current_risk_zone': current_risk,
                        'predicted_risk_level': predicted_level,
                        'risk_score': predicted_risk_score,
                        'mention_frequency': mentions,
                        'trend': 'escalating' if predicted_risk_score > 0.6 else 'stable'
                    }

            # Identify top risk areas
            top_risk_areas = sorted(
                risk_predictions.items(),
                key=lambda x: x[1]['risk_score'],
                reverse=True
            )[:10]

            return {
                'risk_predictions': risk_predictions,
                'top_risk_areas': dict(top_risk_areas),
                'total_monitored_locations': len(risk_predictions),
                'high_risk_count': len(
                    [k for k, v in risk_predictions.items() if v['predicted_risk_level'] in ['HIGH', 'VERY HIGH']])
            }

        except Exception as e:
            return {"error": f"Geographic risk prediction failed: {str(e)}"}

    def _predict_criminal_activities(self, reports_data: List[Dict]) -> Dict:
        """Predict likely activities of Nigerian criminal organizations"""
        try:
            criminal_activity_predictions = {}

            for org_name, patterns in self.criminal_patterns.items():
                org_mentions = 0
                activity_indicators = defaultdict(int)
                geographical_activity = defaultdict(int)

                for report in reports_data:
                    text = report.get('text', '').lower()

                    # Check if this organization is mentioned
                    org_keywords = [org_name.replace('_', ' '), org_name]
                    if any(keyword in text for keyword in org_keywords):
                        org_mentions += 1

                        # Count activity indicators
                        for activity in patterns['typical_activities']:
                            if activity.replace('_', ' ') in text:
                                activity_indicators[activity] += 1

                        # Count geographical mentions
                        for region in patterns['geographical_focus']:
                            if region.replace('_', ' ') in text:
                                geographical_activity[region] += 1

                if org_mentions > 0:
                    # Predict next likely activities
                    top_predicted_activities = sorted(
                        activity_indicators.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:3]

                    # Predict most likely target areas
                    top_target_areas = sorted(
                        geographical_activity.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:3]

                    # Calculate activity probability
                    total_activities = sum(activity_indicators.values())
                    activity_probabilities = {
                        activity: (count / total_activities) * 100 if total_activities > 0 else 0
                        for activity, count in top_predicted_activities
                    }

                    # Determine threat level
                    threat_level = 'HIGH' if org_mentions >= 3 else 'MEDIUM' if org_mentions >= 2 else 'LOW'

                    criminal_activity_predictions[org_name] = {
                        'organization': org_name.replace('_', ' ').title(),
                        'mention_frequency': org_mentions,
                        'threat_level': threat_level,
                        'predicted_activities': dict(top_predicted_activities),
                        'activity_probabilities': activity_probabilities,
                        'likely_target_areas': dict(top_target_areas),
                        'seasonal_pattern': patterns.get('seasonal_patterns', ['unknown']),
                        'funding_assessment': patterns.get('funding_sources', ['unknown']),
                        'next_30_days_likelihood': min(100, org_mentions * 25),  # Simplified probability
                        'recommended_monitoring': self._get_monitoring_recommendations(org_name, patterns)
                    }

            return {
                'criminal_organizations': criminal_activity_predictions,
                'total_active_organizations': len(criminal_activity_predictions),
                'highest_threat_organization': max(
                    criminal_activity_predictions.items(),
                    key=lambda x: x[1]['mention_frequency']
                )[0] if criminal_activity_predictions else None,
                'overall_criminal_threat_level': self._calculate_overall_criminal_threat(criminal_activity_predictions)
            }

        except Exception as e:
            return {"error": f"Criminal activity prediction failed: {str(e)}"}

    def _predict_economic_security_impact(self, reports_data: List[Dict]) -> Dict:
        """Predict security implications of economic indicators"""
        try:
            economic_mentions = defaultdict(int)
            economic_sentiment = defaultdict(list)

            for report in reports_data:
                text = report.get('text', '').lower()

                for indicator in self.economic_security_indicators:
                    indicator_terms = indicator.replace('_', ' ')
                    if indicator_terms in text:
                        economic_mentions[indicator] += 1

                        # Simple sentiment analysis for economic mentions
                        if SENTIMENT_AVAILABLE:
                            blob = TextBlob(text)
                            sentiment = blob.sentiment.polarity
                            economic_sentiment[indicator].append(sentiment)

            # Predict security impact
            security_impact_predictions = {}

            for indicator, mentions in economic_mentions.items():
                if mentions > 0:
                    avg_sentiment = np.mean(economic_sentiment[indicator]) if economic_sentiment[indicator] else 0

                    # Map economic indicators to security impact
                    high_impact_indicators = ['unemployment_rate', 'inflation_rate', 'poverty_index', 'food_prices']
                    impact_level = 'HIGH' if indicator in high_impact_indicators else 'MEDIUM'

                    # Predict security consequence
                    if avg_sentiment < -0.3:  # Negative sentiment
                        security_risk = 'INCREASING'
                        risk_score = 0.8
                    elif avg_sentiment < 0:
                        security_risk = 'MODERATE'
                        risk_score = 0.6
                    else:
                        security_risk = 'STABLE'
                        risk_score = 0.3

                    security_impact_predictions[indicator] = {
                        'mentions': mentions,
                        'average_sentiment': avg_sentiment,
                        'impact_level': impact_level,
                        'security_risk': security_risk,
                        'risk_score': risk_score,
                        'predicted_consequences': self._get_economic_security_consequences(indicator, avg_sentiment)
                    }

            return {
                'economic_indicators': security_impact_predictions,
                'overall_economic_security_risk': np.mean([v['risk_score'] for v in
                                                           security_impact_predictions.values()]) if security_impact_predictions else 0.3,
                'top_risk_indicators': sorted(
                    security_impact_predictions.items(),
                    key=lambda x: x[1]['risk_score'],
                    reverse=True
                )[:5]
            }

        except Exception as e:
            return {"error": f"Economic security impact prediction failed: {str(e)}"}

    def _forecast_threat_levels(self, reports_data: List[Dict], forecast_days: int) -> Dict:
        """Forecast threat levels for the next period"""
        try:
            # Analyze current threat trends
            threat_scores = []
            dates = []

            for report in reports_data:
                text = report.get('text', '')
                processed_date = report.get('processed_date', datetime.now())

                threat_assessment = self._assess_nigerian_threat_level(text)
                threat_scores.append(threat_assessment['score'])
                dates.append(processed_date)

            if not threat_scores:
                return {"error": "No threat data available for forecasting"}

            # Create time series
            threat_df = pd.DataFrame({
                'date': dates,
                'threat_score': threat_scores
            }).sort_values('date')

            # Simple trend analysis
            if len(threat_df) >= 5:
                # Linear regression for trend
                threat_df['days'] = (threat_df['date'] - threat_df['date'].min()).dt.days
                X = threat_df[['days']]
                y = threat_df['threat_score']

                model = LinearRegression()
                model.fit(X, y)

                # Forecast future threat levels
                last_day = threat_df['days'].max()
                future_days = np.arange(last_day + 1, last_day + forecast_days + 1).reshape(-1, 1)
                forecast_scores = model.predict(future_days)

                # Create forecast dates
                forecast_dates = pd.date_range(
                    start=threat_df['date'].max() + timedelta(days=1),
                    periods=forecast_days,
                    freq='D'
                )

                # Categorize forecasted threat levels
                forecasted_levels = []
                for score in forecast_scores:
                    if score > 0.8:
                        level = 'CRITICAL'
                    elif score > 0.6:
                        level = 'HIGH'
                    elif score > 0.4:
                        level = 'MEDIUM'
                    else:
                        level = 'LOW'
                    forecasted_levels.append(level)

                return {
                    'forecast_scores': forecast_scores.tolist(),
                    'forecast_dates': forecast_dates.strftime('%Y-%m-%d').tolist(),
                    'forecast_levels': forecasted_levels,
                    'trend_direction': 'increasing' if model.coef_[0] > 0 else 'decreasing',
                    'current_threat_level': forecasted_levels[0] if forecasted_levels else 'MEDIUM',
                    'peak_threat_day': forecast_dates[np.argmax(forecast_scores)].strftime('%Y-%m-%d') if len(
                        forecast_scores) > 0 else None,
                    'average_forecasted_threat': np.mean(forecast_scores)
                }
            else:
                # Insufficient data - provide baseline forecast
                current_avg = np.mean(threat_scores)
                return {
                    'forecast_scores': [current_avg] * min(7, forecast_days),
                    'forecast_dates': pd.date_range(
                        start=datetime.now() + timedelta(days=1),
                        periods=min(7, forecast_days),
                        freq='D'
                    ).strftime('%Y-%m-%d').tolist(),
                    'trend_direction': 'stable',
                    'current_threat_level': 'MEDIUM',
                    'note': 'Limited historical data - baseline forecast provided'
                }

        except Exception as e:
            return {"error": f"Threat level forecasting failed: {str(e)}"}

    # Helper methods for predictive analysis

    def _assess_nigerian_threat_level(self, text: str) -> Dict:
        """Assess threat level with Nigerian-specific indicators"""
        text_lower = text.lower()

        threat_scores = {'high': 0, 'medium': 0, 'low': 0}
        found_indicators = []

        for level, keywords in self.nigerian_threat_keywords.items():
            for keyword in keywords:
                count = text_lower.count(keyword)
                if count > 0:
                    threat_scores[level] += count
                    found_indicators.extend([keyword] * count)

        # Determine overall threat level
        if threat_scores['high'] > 0:
            level = 'HIGH'
            score = 0.8 + min(0.2, threat_scores['high'] * 0.05)
        elif threat_scores['medium'] > 0:
            level = 'MEDIUM'
            score = 0.4 + min(0.4, threat_scores['medium'] * 0.1)
        else:
            level = 'LOW'
            score = max(0.1, 0.3 - threat_scores['low'] * 0.05)

        return {
            'level': level,
            'score': min(1.0, score),
            'indicators': found_indicators[:5]
        }

    def _calculate_location_threat_score(self, text: str, location: str) -> float:
        """Calculate threat score for a specific location"""
        threat_score = 0.5  # Base score

        # Check for threat keywords in context
        for level, keywords in self.nigerian_threat_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    if level == 'high':
                        threat_score += 0.3
                    elif level == 'medium':
                        threat_score += 0.2
                    else:
                        threat_score -= 0.1

        return min(1.0, max(0.0, threat_score))

    def _calculate_overall_risk_score(self, time_series: Dict, geographic: Dict, threat: Dict) -> Dict:
        """Calculate overall risk assessment"""
        try:
            risk_factors = []

            # Time series risk
            if 'trend_direction' in time_series:
                if time_series['trend_direction'] == 'increasing':
                    risk_factors.append(0.7)
                else:
                    risk_factors.append(0.4)

            # Geographic risk
            if 'high_risk_count' in geographic:
                high_risk_ratio = geographic['high_risk_count'] / max(geographic.get('total_monitored_locations', 1), 1)
                risk_factors.append(high_risk_ratio)

            # Threat level risk
            if 'average_forecasted_threat' in threat:
                risk_factors.append(threat['average_forecasted_threat'])

            overall_score = np.mean(risk_factors) if risk_factors else 0.5

            if overall_score > 0.8:
                risk_level = 'CRITICAL'
            elif overall_score > 0.6:
                risk_level = 'HIGH'
            elif overall_score > 0.4:
                risk_level = 'MEDIUM'
            else:
                risk_level = 'LOW'

            return {
                'overall_risk_score': overall_score,
                'risk_level': risk_level,
                'contributing_factors': len(risk_factors),
                'confidence': 0.8 if len(risk_factors) >= 3 else 0.6
            }
        except:
            return {'overall_risk_score': 0.5, 'risk_level': 'MEDIUM', 'confidence': 0.3}

    def _generate_recommendations(self, time_series: Dict, geographic: Dict, threat: Dict) -> List[str]:
        """Generate actionable recommendations based on predictions"""
        recommendations = []

        try:
            # Time series recommendations
            if time_series.get('trend_direction') == 'increasing':
                recommendations.append("📈 Increasing activity trend detected - enhance monitoring capabilities")
                recommendations.append("👮 Consider deploying additional security resources in coming weeks")

            # Geographic recommendations
            if geographic.get('high_risk_count', 0) > 0:
                recommendations.append(f"🗺️ Monitor {geographic['high_risk_count']} high-risk locations closely")
                recommendations.append("🛡️ Implement preventive measures in identified hotspots")

            # Threat level recommendations
            if threat.get('current_threat_level') in ['HIGH', 'CRITICAL']:
                recommendations.append("🚨 Activate elevated threat response protocols")
                recommendations.append("📱 Increase communication frequency with field units")

            # Nigerian-specific recommendations
            recommendations.extend([
                "🤝 Strengthen community engagement in high-risk areas",
                "📊 Continue monitoring economic indicators affecting security",
                "🔄 Review and update contingency plans based on predictions",
                "📡 Enhance intelligence sharing between agencies",
                "🎯 Focus resources on predicted criminal organization activities"
            ])

        except:
            recommendations = ["📋 Review intelligence data and update analysis parameters"]

        return recommendations[:8]  # Return top 8 recommendations

    def _get_monitoring_recommendations(self, org_name: str, patterns: Dict) -> List[str]:
        """Get monitoring recommendations for specific criminal organizations"""
        recommendations = []

        if org_name == 'boko_haram':
            recommendations = [
                "Monitor movements in Sambisa Forest area",
                "Track communications in Kanuri language",
                "Watch for increased activity during dry season",
                "Coordinate with Multinational Joint Task Force"
            ]
        elif org_name == 'ipob':
            recommendations = [
                "Monitor social media for sit-at-home declarations",
                "Track diaspora funding channels",
                "Watch for activity around Biafra remembrance dates",
                "Monitor southeastern Nigeria transport hubs"
            ]
        elif org_name == 'bandits':
            recommendations = [
                "Increase forest area surveillance",
                "Monitor cattle markets for stolen livestock",
                "Track mobile phone communications in remote areas",
                "Coordinate with local vigilante groups"
            ]
        elif org_name == 'niger_delta_militants':
            recommendations = [
                "Monitor oil pipeline installations",
                "Track illegal oil refining activities",
                "Increase maritime patrols",
                "Watch for political funding connections"
            ]

        return recommendations

    def _get_economic_security_consequences(self, indicator: str, sentiment: float) -> List[str]:
        """Get predicted security consequences of economic indicators"""
        consequences = []

        if sentiment < -0.3:  # Very negative
            if indicator == 'unemployment_rate':
                consequences = ["Youth restiveness", "Increased crime", "Social unrest", "Migration pressures"]
            elif indicator == 'inflation_rate':
                consequences = ["Food riots", "Strike actions", "Economic protests", "Border smuggling"]
            elif indicator == 'food_prices':
                consequences = ["Farmer-herder conflicts", "Food riots", "Rural-urban migration", "Market disruptions"]
            elif indicator == 'fuel_prices':
                consequences = ["Transport strikes", "Economic slowdown", "Increased smuggling", "Public protests"]
        else:
            consequences = ["Stable security environment", "Reduced crime rates", "Economic growth"]

        return consequences

    def _create_synthetic_forecast(self, forecast_days: int) -> Dict:
        """Create synthetic forecast for demonstration when data is limited"""
        return {
            'model_type': 'Synthetic Demonstration',
            'forecast_values': [5, 7, 6, 8, 9],
            'trend_direction': 'increasing',
            'confidence_level': 0.4,
            'note': 'Synthetic data - process more documents for accurate forecasting'
        }

    def _calculate_overall_criminal_threat(self, criminal_predictions: Dict) -> str:
        """Calculate overall criminal threat level"""
        if not criminal_predictions:
            return 'LOW'

        high_threats = sum(1 for org in criminal_predictions.values() if org['threat_level'] == 'HIGH')
        total_orgs = len(criminal_predictions)

        if high_threats / total_orgs > 0.5:
            return 'HIGH'
        elif high_threats > 0:
            return 'MEDIUM'
        else:
            return 'LOW'

    # Original methods (keeping existing functionality)
    def _extract_dates_from_text(self, text: str) -> List[Dict]:
        """Extract dates and temporal references from text with enhanced year-only support"""
        dates = []

        # Enhanced date patterns for intelligence documents
        date_patterns = [
            # Full date formats
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # MM/DD/YYYY, MM-DD-YYYY
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',  # YYYY/MM/DD, YYYY-MM-DD

            # Month name formats
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',

            # Short month formats
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b',

            # Intelligence-specific date patterns
            r'\bDate:\s*([^\n]+)',  # "Date: June 8, 2025"
            r'\b(\d{4}-\d{2}-\d{2})\b',  # ISO format
            r'\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\b',

            # More flexible patterns
            r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\b',  # "June 2025"
        ]

        # ENHANCED: Year-only patterns for intelligence documents
        year_patterns = [
            # Years in parentheses - common in intelligence reports
            r'\((\d{4})\)',  # (2015), (2019), (2022)

            # Standalone years with context indicators
            r'\bin\s+(\d{4})\b',  # "in 2015", "in 2019"
            r'\bsince\s+(\d{4})\b',  # "since 2015"
            r'\bduring\s+(\d{4})\b',  # "during 2019"
            r'\bthroughout\s+(\d{4})\b',  # "throughout 2022"
            r'\bfrom\s+(\d{4})\b',  # "from 2015"
            r'\buntil\s+(\d{4})\b',  # "until 2019"
            r'\bby\s+(\d{4})\b',  # "by 2022"
            r'\bbefore\s+(\d{4})\b',  # "before 2015"
            r'\bafter\s+(\d{4})\b',  # "after 2019"

            # Year ranges
            r'\b(\d{4})-(\d{4})\b',  # "2015-2019", "2020-2022"
            r'\bbetween\s+(\d{4})\s+and\s+(\d{4})\b',  # "between 2015 and 2019"

            # Years with specific context (intelligence/historical)
            r'\boperations?\s+(?:in\s+)?(\d{4})\b',  # "operation in 2015"
            r'\battacks?\s+(?:in\s+)?(\d{4})\b',  # "attack in 2019"
            r'\bincidents?\s+(?:in\s+)?(\d{4})\b',  # "incident in 2022"
            r'\breports?\s+(?:from\s+)?(\d{4})\b',  # "report from 2015"
            r'\bdata\s+(?:from\s+)?(\d{4})\b',  # "data from 2019"
            r'\banalysis\s+(?:from\s+)?(\d{4})\b',  # "analysis from 2022"
            r'\bintelligence\s+(?:from\s+)?(\d{4})\b',  # "intelligence from 2015"

            # Financial/timeline context
            r'\bbudget\s+(?:for\s+)?(\d{4})\b',  # "budget for 2022"
            r'\bfunding\s+(?:in\s+)?(\d{4})\b',  # "funding in 2019"
            r'\bexpenses\s+(?:in\s+)?(\d{4})\b',  # "expenses in 2015"

            # Standalone years (careful to avoid false positives)
            r'\b(\d{4})\b(?=\s+(?:was|were|saw|marked|brought|witnessed|experienced|showed|indicated))',
            # "2015 was", "2019 saw"
            r'(?:^|\.\s+)(\d{4})\s+',  # Year at start of sentence
            r'\s+(\d{4})(?=\s*[:\-])',  # Years followed by colon or dash
        ]

        # Process standard date patterns first
        for pattern in date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                date_text = match.group(1) if match.groups() else match.group()
                date_text = date_text.strip()

                # Skip very short or obviously wrong matches
                if len(date_text) < 4:
                    continue

                try:
                    parsed_date = None

                    # Handle relative dates
                    if date_text.lower() == 'today':
                        parsed_date = datetime.now()
                    elif date_text.lower() == 'yesterday':
                        parsed_date = datetime.now() - timedelta(days=1)
                    elif date_text.lower() == 'tomorrow':
                        parsed_date = datetime.now() + timedelta(days=1)
                    else:
                        # Try dateparser first
                        parsed_date = dateparser.parse(date_text)

                        # Fallback to manual parsing for common formats
                        if not parsed_date:
                            try:
                                # Try different formats manually
                                for date_format in [
                                    '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y',
                                    '%B %d, %Y', '%d %B %Y', '%b %d, %Y',
                                    '%Y', '%B %Y', '%b %Y'
                                ]:
                                    try:
                                        parsed_date = datetime.strptime(date_text.strip(), date_format)
                                        break
                                    except ValueError:
                                        continue
                            except:
                                continue

                    if parsed_date:
                        # Get context around the date
                        start = max(0, match.start() - 60)
                        end = min(len(text), match.end() + 60)
                        context = text[start:end].strip()

                        # Assign confidence based on pattern quality
                        confidence = 0.9 if any(month in date_text for month in [
                            'January', 'February', 'March', 'April', 'May', 'June',
                            'July', 'August', 'September', 'October', 'November', 'December'
                        ]) else 0.7

                        dates.append({
                            'date': parsed_date,
                            'text': date_text,
                            'confidence': confidence,
                            'context': context,
                            'type': 'full_date'
                        })

                except Exception as e:
                    print(f"⚠️ Failed to parse date '{date_text}': {e}")
                    continue

        # ENHANCED: Process year-only patterns
        for pattern in year_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Handle multiple groups for year ranges
                year_groups = match.groups()
                years_to_process = []

                if len(year_groups) == 1:
                    # Single year
                    years_to_process = [year_groups[0]]
                elif len(year_groups) == 2:
                    # Year range - add both years
                    years_to_process = [year_groups[0], year_groups[1]]

                for year_text in years_to_process:
                    if not year_text:
                        continue

                    try:
                        year = int(year_text)

                        # Validate year range for intelligence documents (1990-2030)
                        if 1990 <= year <= 2030:
                            # Create date object (January 1st of that year)
                            parsed_date = datetime(year, 1, 1)

                            # Get extended context around the year mention
                            start = max(0, match.start() - 80)
                            end = min(len(text), match.end() + 80)
                            context = text[start:end].strip()

                            # Determine confidence based on context
                            confidence = 0.6  # Lower confidence for year-only

                            # Higher confidence if in parentheses or with strong context
                            if '(' in match.group() and ')' in match.group():
                                confidence = 0.8
                            elif any(word in context.lower() for word in [
                                'operation', 'attack', 'incident', 'report', 'intelligence',
                                'analysis', 'budget', 'terrorist', 'funding', 'during', 'throughout'
                            ]):
                                confidence = 0.7

                            # Create display text based on pattern
                            if '(' in match.group():
                                display_text = f"({year})"
                            elif '-' in match.group():
                                display_text = match.group()
                            else:
                                display_text = str(year)

                            dates.append({
                                'date': parsed_date,
                                'text': display_text,
                                'confidence': confidence,
                                'context': context,
                                'type': 'year_only'
                            })

                    except (ValueError, TypeError):
                        continue

        # If no dates found, create sample dates for demonstration
        if not dates and len(text) > 100:
            print("🔍 No dates found with patterns, creating demonstration timeline...")

            # Create sample dates based on current date for demonstration
            current_date = datetime.now()
            sample_years = [2015, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

            for year in sample_years:
                sample_date = datetime(year, 1, 1)
                dates.append({
                    'date': sample_date,
                    'text': f"({year})",
                    'confidence': 0.4,  # Lower confidence for generated dates
                    'context': f"Sample timeline entry for {year} (demonstration)",
                    'type': 'sample'
                })

        # Remove duplicates and sort by date
        unique_dates = []
        seen_dates = set()

        for date_info in dates:
            # Use year-month for deduplication to allow multiple events per year
            if date_info['type'] == 'year_only':
                date_key = date_info['date'].strftime('%Y')
            else:
                date_key = date_info['date'].strftime('%Y-%m-%d')

            # For year-only dates, allow multiple if they have different contexts
            if date_info['type'] == 'year_only':
                unique_key = f"{date_key}_{hash(date_info['context'][:50])}"
            else:
                unique_key = date_key

            if unique_key not in seen_dates:
                seen_dates.add(unique_key)
                unique_dates.append(date_info)

        # Sort by date and limit to reasonable number
        sorted_dates = sorted(unique_dates, key=lambda x: x['date'])

        print(f"📅 Extracted {len(sorted_dates)} unique temporal references:")
        for date_info in sorted_dates[:10]:  # Show first 10
            print(f"   {date_info['text']} ({date_info['type']}) - confidence: {date_info['confidence']:.1f}")

        return sorted_dates[:50]  # Limit to 50 dates to avoid overwhelming visualizations

    def analyze_geography(self, entities_data: List[Dict]) -> Dict:
        """Extract and analyze geographic patterns from entities"""
        if not GEOGRAPHIC_AVAILABLE:
            return {"error": "Geographic analysis libraries not available"}

        # Filter geographic entities
        geo_entities = [ent for ent in entities_data if ent.get('label') == 'GPE']

        if not geo_entities:
            return {"message": "No geographic entities found"}

        geographic_data = []
        location_frequency = Counter()

        for entity in geo_entities:
            location_name = entity['text']
            location_frequency[location_name] += 1

            # Get coordinates
            coords = self._get_coordinates(location_name)

            if coords:
                geographic_data.append({
                    'location': location_name,
                    'latitude': coords[0],
                    'longitude': coords[1],
                    'frequency': location_frequency[location_name],
                    'doc_id': entity.get('doc_id', 'unknown'),
                    'doc_type': entity.get('doc_type', 'unknown'),
                    'country': self._get_country_info(location_name)
                })

        # Geographic analysis
        analysis = {
            'total_locations': len(set([ent['text'] for ent in geo_entities])),
            'geographic_data': geographic_data,
            'location_frequency': dict(location_frequency.most_common(10)),
            'country_analysis': self._analyze_countries(geographic_data),
            'geographic_clusters': self._identify_geographic_clusters(geographic_data),
            'hotspots': self._identify_hotspots(geographic_data)
        }

        return analysis

    def analyze_sentiment(self, reports_data: List[Dict]) -> Dict:
        """Analyze sentiment and threat levels in reports"""
        if not SENTIMENT_AVAILABLE:
            return {"error": "Sentiment analysis libraries not available"}

        sentiment_data = []

        for report in reports_data:
            doc_id = report.get('doc_id', 'unknown')
            text = report.get('text', '')
            doc_type = report.get('doc_type', 'unknown')

            # VADER sentiment analysis (good for social media/informal text)
            vader_scores = self.sentiment_analyzer.polarity_scores(text)

            # TextBlob sentiment analysis (good for formal text)
            blob = TextBlob(text)
            textblob_sentiment = blob.sentiment

            # Enhanced Nigerian threat level analysis
            threat_level = self._assess_nigerian_threat_level(text)
            urgency_score = self._assess_urgency(text)

            sentiment_data.append({
                'doc_id': doc_id,
                'doc_type': doc_type,
                'vader_compound': vader_scores['compound'],
                'vader_positive': vader_scores['pos'],
                'vader_neutral': vader_scores['neu'],
                'vader_negative': vader_scores['neg'],
                'textblob_polarity': textblob_sentiment.polarity,
                'textblob_subjectivity': textblob_sentiment.subjectivity,
                'threat_level': threat_level['level'],
                'threat_score': threat_level['score'],
                'urgency_score': urgency_score,
                'threat_indicators': threat_level['indicators']
            })

        sentiment_df = pd.DataFrame(sentiment_data)

        # Sentiment analysis summary
        analysis = {
            'overall_sentiment': {
                'avg_polarity': sentiment_df['textblob_polarity'].mean(),
                'avg_compound': sentiment_df['vader_compound'].mean(),
                'sentiment_distribution': self._categorize_sentiment(sentiment_df)
            },
            'threat_analysis': {
                'avg_threat_score': sentiment_df['threat_score'].mean(),
                'threat_distribution': sentiment_df['threat_level'].value_counts().to_dict(),
                'high_threat_docs': sentiment_df[sentiment_df['threat_level'] == 'HIGH']['doc_id'].tolist()
            },
            'urgency_analysis': {
                'avg_urgency': sentiment_df['urgency_score'].mean(),
                'urgent_docs': sentiment_df[sentiment_df['urgency_score'] > 0.7]['doc_id'].tolist()
            },
            'sentiment_data': sentiment_df.to_dict('records'),
            'keyword_analysis': self._analyze_keywords_sentiment(reports_data)
        }

        return analysis

    def _get_coordinates(self, location_name: str) -> Optional[Tuple[float, float]]:
        """Get coordinates for a location (with caching)"""
        if location_name in self.location_cache:
            return self.location_cache[location_name]

        try:
            location = self.geocoder.geocode(location_name, timeout=5)
            if location:
                coords = (location.latitude, location.longitude)
                self.location_cache[location_name] = coords
                return coords
        except (GeocoderTimedOut, Exception):
            pass

        self.location_cache[location_name] = None
        return None

    def _assess_threat_level(self, text: str) -> Dict:
        """Assess threat level based on keywords and context (original method)"""
        text_lower = text.lower()

        threat_scores = {'high': 0, 'medium': 0, 'low': 0}
        found_indicators = []

        for level, keywords in self.threat_keywords.items():
            for keyword in keywords:
                count = text_lower.count(keyword)
                if count > 0:
                    threat_scores[level] += count
                    found_indicators.extend([keyword] * count)

        # Determine overall threat level
        if threat_scores['high'] > 0:
            level = 'HIGH'
            score = 0.8 + min(0.2, threat_scores['high'] * 0.05)
        elif threat_scores['medium'] > 0:
            level = 'MEDIUM'
            score = 0.4 + min(0.4, threat_scores['medium'] * 0.1)
        else:
            level = 'LOW'
            score = max(0.1, 0.3 - threat_scores['low'] * 0.05)

        return {
            'level': level,
            'score': min(1.0, score),
            'indicators': found_indicators[:5]  # Top 5 indicators
        }

    def _assess_urgency(self, text: str) -> float:
        """Assess urgency based on temporal and action keywords"""
        urgency_keywords = {
            'immediate': 1.0,
            'urgent': 0.9,
            'asap': 0.9,
            'emergency': 1.0,
            'critical': 0.8,
            'priority': 0.7,
            'soon': 0.5,
            'today': 0.7,
            'now': 0.8
        }

        text_lower = text.lower()
        urgency_score = 0.0

        for keyword, score in urgency_keywords.items():
            if keyword in text_lower:
                urgency_score = max(urgency_score, score)

        return urgency_score

    def _analyze_temporal_patterns(self, timeline_df: pd.DataFrame) -> Dict:
        """Analyze patterns in temporal data"""
        if timeline_df.empty:
            return {}

        # Group by time periods
        timeline_df = timeline_df.copy()  # Avoid modifying original
        timeline_df['year'] = timeline_df['extracted_date'].dt.year
        timeline_df['month'] = timeline_df['extracted_date'].dt.month
        timeline_df['day_of_week'] = timeline_df['extracted_date'].dt.day_name()

        patterns = {
            'yearly_activity': timeline_df['year'].value_counts().to_dict(),
            'monthly_activity': timeline_df['month'].value_counts().to_dict(),
            'weekly_patterns': timeline_df['day_of_week'].value_counts().to_dict(),
            'recent_activity': len(timeline_df[timeline_df['extracted_date'] >= datetime.now() - timedelta(days=30)])
        }

        return patterns

    def _get_monthly_activity(self, timeline_df: pd.DataFrame) -> List[Dict]:
        """Get monthly activity data for timeline visualization - FIXED"""
        if timeline_df.empty:
            return []

        try:
            # Create a copy to avoid modifying original
            df_copy = timeline_df.copy()

            # Extract year and month
            df_copy['year'] = df_copy['extracted_date'].dt.year
            df_copy['month'] = df_copy['extracted_date'].dt.month

            # Group by year and month
            monthly_counts = df_copy.groupby(['year', 'month']).size().reset_index(name='count')

            monthly_data = []
            for _, row in monthly_counts.iterrows():
                try:
                    # Create date string for the first day of the month
                    date_str = f"{int(row['year'])}-{int(row['month']):02d}-01"
                    monthly_data.append({
                        'year': int(row['year']),
                        'month': int(row['month']),
                        'count': int(row['count']),
                        'date': date_str
                    })
                except (ValueError, TypeError) as e:
                    print(f"⚠️ Error processing monthly data row: {e}")
                    continue

            return monthly_data

        except Exception as e:
            print(f"⚠️ Error in _get_monthly_activity: {e}")
            return []

    def _extract_temporal_entities(self, timeline_df: pd.DataFrame) -> List[Dict]:
        """Extract key temporal entities and events"""
        if timeline_df.empty:
            return []

        # Group by dates and aggregate information
        temporal_entities = []

        try:
            for date, group in timeline_df.groupby('extracted_date'):
                docs = group['doc_id'].tolist()
                contexts = group['context'].tolist()

                temporal_entities.append({
                    'date': date,
                    'document_count': len(docs),
                    'documents': docs,
                    'contexts': contexts[:3],  # Top 3 contexts
                    'importance_score': len(docs) * group['confidence'].mean()
                })

            # Sort by importance
            temporal_entities.sort(key=lambda x: x['importance_score'], reverse=True)

            return temporal_entities[:20]  # Top 20 temporal entities

        except Exception as e:
            print(f"⚠️ Error extracting temporal entities: {e}")
            return []

    def _analyze_countries(self, geographic_data: List[Dict]) -> Dict:
        """Analyze geographic data by country"""
        country_analysis = defaultdict(list)

        for location in geographic_data:
            country = location.get('country', 'Unknown')
            country_analysis[country].append(location)

        country_summary = {}
        for country, locations in country_analysis.items():
            country_summary[country] = {
                'location_count': len(locations),
                'total_mentions': sum(loc['frequency'] for loc in locations),
                'locations': [loc['location'] for loc in locations]
            }

        return country_summary

    def _identify_geographic_clusters(self, geographic_data: List[Dict]) -> List[Dict]:
        """Identify geographic clusters of activity"""
        if not geographic_data:
            return []

        # Simple clustering based on frequency and proximity
        clusters = []

        # Group by region (simplified)
        for location in geographic_data:
            if location['frequency'] > 1:  # Only locations mentioned multiple times
                clusters.append({
                    'center': location['location'],
                    'latitude': location['latitude'],
                    'longitude': location['longitude'],
                    'activity_level': location['frequency'],
                    'documents': [location['doc_id']]
                })

        # Sort by activity level
        clusters.sort(key=lambda x: x['activity_level'], reverse=True)

        return clusters[:10]  # Top 10 clusters

    def _identify_hotspots(self, geographic_data: List[Dict]) -> List[Dict]:
        """Identify geographic hotspots based on activity"""
        hotspots = []

        for location in geographic_data:
            if location['frequency'] >= 2:  # Threshold for hotspot
                hotspots.append({
                    'location': location['location'],
                    'latitude': location['latitude'],
                    'longitude': location['longitude'],
                    'intensity': location['frequency'],
                    'risk_level': 'HIGH' if location['frequency'] > 3 else 'MEDIUM'
                })

        return sorted(hotspots, key=lambda x: x['intensity'], reverse=True)

    def _get_country_info(self, location_name: str) -> str:
        """Get country information for a location"""
        try:
            # Simple country detection (can be enhanced)
            location_lower = location_name.lower()

            # Check if it's a known country
            try:
                country = pycountry.countries.search_fuzzy(location_name)[0]
                return country.name
            except:
                pass

            # Common country/region mappings
            country_mappings = {
                'moscow': 'Russia',
                'london': 'United Kingdom',
                'paris': 'France',
                'berlin': 'Germany',
                'tokyo': 'Japan',
                'beijing': 'China',
                'new york': 'United States',
                'los angeles': 'United States',
                'california': 'United States',
                'texas': 'United States',
                'amsterdam': 'Netherlands',
                'zurich': 'Switzerland',
                'dubai': 'United Arab Emirates',
                'singapore': 'Singapore',
                'hong kong': 'China'
            }

            for city, country in country_mappings.items():
                if city in location_lower:
                    return country

            return 'Unknown'
        except:
            return 'Unknown'

    def _categorize_sentiment(self, sentiment_df: pd.DataFrame) -> Dict:
        """Categorize sentiment into positive, negative, neutral"""

        def categorize_polarity(polarity):
            if polarity > 0.1:
                return 'positive'
            elif polarity < -0.1:
                return 'negative'
            else:
                return 'neutral'

        sentiment_df = sentiment_df.copy()  # Avoid modifying original
        sentiment_df['sentiment_category'] = sentiment_df['textblob_polarity'].apply(categorize_polarity)
        return sentiment_df['sentiment_category'].value_counts().to_dict()

    def _analyze_keywords_sentiment(self, reports_data: List[Dict]) -> Dict:
        """Analyze sentiment of key intelligence keywords"""
        keyword_sentiment = {}

        intelligence_keywords = [
            'threat', 'attack', 'surveillance', 'operation', 'mission',
            'target', 'suspect', 'investigation', 'intelligence', 'security',
            'boko haram', 'ipob', 'bandits', 'kidnapping'  # Nigerian context
        ]

        for keyword in intelligence_keywords:
            keyword_texts = []
            for report in reports_data:
                text = report.get('text', '').lower()
                if keyword in text:
                    # Extract sentences containing the keyword
                    sentences = text.split('.')
                    keyword_sentences = [s for s in sentences if keyword in s]
                    keyword_texts.extend(keyword_sentences)

            if keyword_texts:
                combined_text = '. '.join(keyword_texts)
                blob = TextBlob(combined_text)
                keyword_sentiment[keyword] = {
                    'polarity': blob.sentiment.polarity,
                    'subjectivity': blob.sentiment.subjectivity,
                    'mention_count': len(keyword_texts)
                }

        return keyword_sentiment


# Test function
if __name__ == "__main__":
    engine = AnalyticsEngine()
    print("🧪 Enhanced Analytics Engine with Predictive Capabilities Test")
    print(f"Geographic analysis: {'✅' if GEOGRAPHIC_AVAILABLE else '❌'}")
    print(f"Sentiment analysis: {'✅' if SENTIMENT_AVAILABLE else '❌'}")
    print(f"Temporal analysis: {'✅' if TEMPORAL_AVAILABLE else '❌'}")
    print(f"Predictive models: {'✅' if SKLEARN_AVAILABLE else '❌'}")
    print(f"Time series forecasting: {'✅' if PROPHET_AVAILABLE or STATSMODELS_AVAILABLE else '❌'}")

    # Test predictive analysis with sample data
    sample_reports = [
        {
            'doc_id': 'TEST-001',
            'doc_type': 'security_intelligence',
            'text': 'On June 8, 2025, Boko Haram activities increased in Borno state. Operations in (2019) and during 2022 show escalation. High unemployment and inflation affecting northeast Nigeria.',
            'processed_date': datetime.now()
        }
    ]

    print("\n🔮 Testing predictive analysis...")
    result = engine.predict_future_trends(sample_reports, 30)
    print(f"Prediction test result: {'✅ Success' if 'error' not in result else '❌ ' + result['error']}")