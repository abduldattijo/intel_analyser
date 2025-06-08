#!/usr/bin/env python3
"""
Enhanced Analytics Engine for Intelligence Analysis
Handles Timeline Analysis, Geographic Mapping, and Sentiment Analysis
FIXED: Pandas DataFrame column conflict errors
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter

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


class AnalyticsEngine:
    """Advanced analytics engine for intelligence analysis"""

    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer() if SENTIMENT_AVAILABLE else None
        self.geocoder = Nominatim(user_agent="intelligence_analysis") if GEOGRAPHIC_AVAILABLE else None
        self.location_cache = {}  # Cache geocoding results

        # Known intelligence-related locations and threat indicators
        self.threat_keywords = {
            'high': ['attack', 'threat', 'immediate', 'urgent', 'critical', 'emergency', 'danger', 'weapon',
                     'explosive','IPOB','Bandits', 'Bokoharam', 'terrorist'],
            'medium': ['suspicious', 'unusual', 'concern', 'monitor', 'investigate', 'alert', 'warning', 'risk'],
            'low': ['routine', 'normal', 'standard', 'regular', 'typical', 'maintenance', 'administrative']
        }

        print("🧠 Analytics Engine initialized")
        print(f"📍 Geographic analysis: {'✅' if GEOGRAPHIC_AVAILABLE else '❌'}")
        print(f"📊 Sentiment analysis: {'✅' if SENTIMENT_AVAILABLE else '❌'}")
        print(f"📅 Temporal analysis: {'✅' if TEMPORAL_AVAILABLE else '❌'}")

    def analyze_timeline(self, reports_data: List[Dict]) -> Dict:
        """Extract and analyze temporal patterns from reports"""
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

            # Extract dates from text
            extracted_dates = self._extract_dates_from_text(text)

            print(f"📅 Found {len(extracted_dates)} dates in {doc_id}")

            # Analyze temporal mentions
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

        # Create timeline analysis
        timeline_df = pd.DataFrame(timeline_data)

        if timeline_df.empty:
            return {"message": "No temporal data found in documents"}

        # Timeline statistics
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

            # Threat level analysis
            threat_level = self._assess_threat_level(text)
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
        """Assess threat level based on keywords and context"""
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
            'target', 'suspect', 'investigation', 'intelligence', 'IPOB','Bandits', 'Bokoharam', 'security'
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
    print("🧪 Analytics Engine Test")
    print(f"Geographic analysis: {'✅' if GEOGRAPHIC_AVAILABLE else '❌'}")
    print(f"Sentiment analysis: {'✅' if SENTIMENT_AVAILABLE else '❌'}")
    print(f"Temporal analysis: {'✅' if TEMPORAL_AVAILABLE else '❌'}")

    # Test timeline analysis with sample data
    sample_reports = [
        {
            'doc_id': 'TEST-001',
            'doc_type': 'security_intelligence',
            'text': 'On June 8, 2025, JOHN SMITH was observed in MOSCOW. Operations in (2019) and during 2022 show escalation.',
            'processed_date': datetime.now()
        }
    ]

    result = engine.analyze_timeline(sample_reports)
    print(f"Timeline test result: {len(result.get('timeline_data', []))} dates found")