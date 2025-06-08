#!/usr/bin/env python3
"""
Enhanced Intelligence Processor with Text Storage for Analytics
Extracts entities, relationships, and stores original text for temporal analysis
"""

import spacy
import pandas as pd
import networkx as nx
from datetime import datetime
from typing import List, Dict, Tuple, Set
from collections import defaultdict, Counter
import re


class IntelligenceProcessor:
    """Enhanced intelligence processor with analytics support"""

    def __init__(self):
        print("🧠 Initializing Intelligence Processor...")

        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy model loaded successfully")
        except OSError:
            print("❌ spaCy model not found. Please install: python -m spacy download en_core_web_sm")
            raise

        # Entity types to extract
        self.entity_types = {
            'PERSON', 'ORG', 'GPE', 'MONEY', 'DATE', 'TIME',
            'PRODUCT', 'EVENT', 'FAC', 'NORP', 'LOC'
        }

        # Storage for processed data
        self.entities = []
        self.relationships = []
        self.reports = []
        self.document_texts = {}  # NEW: Store original text for analytics

        # Intelligence-specific patterns
        self.intel_patterns = self._load_intelligence_patterns()

        print("🔍 Intelligence Processor ready")

    def _load_intelligence_patterns(self) -> Dict:
        """Load intelligence-specific entity patterns"""
        return {
            'code_names': r'\b[A-Z]{2,}[-_]?[A-Z]{2,}\b',  # OPERATION-NAME, CODE_WORD
            'coordinates': r'\d{1,3}\.\d+[°]?\s*[NSEW]',
            'phone_numbers': r'\+?\d{1,3}[-.\s]?\d{3,4}[-.\s]?\d{3,4}[-.\s]?\d{3,5}',
            'account_numbers': r'\b\d{8,20}\b',
            'crypto_addresses': r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b',
            'ip_addresses': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        }

    def process_document(self, text: str, doc_id: str, doc_type: str) -> List[Dict]:
        """Process a document and extract intelligence entities and relationships"""

        # Store the original text for analytics
        self.document_texts[doc_id] = text

        # Extract entities using spaCy
        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            if ent.label_ in self.entity_types:
                entity = {
                    'text': ent.text.strip(),
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'doc_id': doc_id,
                    'doc_type': doc_type,
                    'confidence': 1.0  # spaCy doesn't provide confidence scores by default
                }
                entities.append(entity)

        # Extract custom intelligence patterns
        custom_entities = self._extract_custom_patterns(text, doc_id, doc_type)
        entities.extend(custom_entities)

        # Extract relationships
        relationships = self._extract_relationships(entities, text)

        # Store entities and relationships
        self.entities.extend(entities)
        self.relationships.extend(relationships)

        # Create report summary
        report = {
            'doc_id': doc_id,
            'doc_type': doc_type,
            'entity_count': len(entities),
            'text_length': len(text),
            'word_count': len(text.split()),
            'processed_date': datetime.now(),
            'entities': entities,
            'relationships': relationships
        }

        self.reports.append(report)

        print(f"📊 Processed {doc_id}: {len(entities)} entities, {len(relationships)} relationships")

        return entities

    def _extract_custom_patterns(self, text: str, doc_id: str, doc_type: str) -> List[Dict]:
        """Extract custom intelligence patterns not caught by spaCy"""
        custom_entities = []

        for pattern_type, pattern in self.intel_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                entity = {
                    'text': match.group().strip(),
                    'label': 'MISC',  # Custom label for intelligence patterns
                    'start': match.start(),
                    'end': match.end(),
                    'doc_id': doc_id,
                    'doc_type': doc_type,
                    'confidence': 0.8,
                    'pattern_type': pattern_type
                }
                custom_entities.append(entity)

        return custom_entities

    def _extract_relationships(self, entities: List[Dict], text: str) -> List[Dict]:
        """Extract relationships between entities"""
        relationships = []

        # Simple proximity-based relationship extraction
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i + 1:], i + 1):
                # Calculate distance between entities
                distance = abs(entity1['start'] - entity2['start'])

                # If entities are close (within 100 characters), consider them related
                if distance < 100:
                    relationship_strength = max(0.1, 1.0 - (distance / 100.0))

                    # Extract context around the relationship
                    start_pos = min(entity1['start'], entity2['start']) - 20
                    end_pos = max(entity1['end'], entity2['end']) + 20
                    context = text[max(0, start_pos):min(len(text), end_pos)]

                    relationship = {
                        'entity1': entity1['text'],
                        'entity2': entity2['text'],
                        'strength': relationship_strength,
                        'context': context.strip(),
                        'doc_id': entity1['doc_id'],
                        'doc_type': entity1['doc_type']
                    }
                    relationships.append(relationship)

        return relationships

    def get_entity_network(self, min_weight: float = 0.1) -> nx.Graph:
        """Create a network graph of entity relationships"""
        G = nx.Graph()

        # Count entity frequencies
        entity_counts = Counter(ent['text'] for ent in self.entities)

        # Add nodes with attributes
        for entity_text, frequency in entity_counts.items():
            # Find entity type
            entity_type = 'UNKNOWN'
            for ent in self.entities:
                if ent['text'] == entity_text:
                    entity_type = ent['label']
                    break

            G.add_node(entity_text, frequency=frequency, entity_type=entity_type)

        # Add edges based on relationships
        relationship_weights = defaultdict(float)

        for rel in self.relationships:
            entity1, entity2 = rel['entity1'], rel['entity2']
            if entity1 in G.nodes() and entity2 in G.nodes():
                relationship_weights[(entity1, entity2)] += rel['strength']

        # Add edges that meet minimum weight threshold
        for (entity1, entity2), weight in relationship_weights.items():
            if weight >= min_weight:
                G.add_edge(entity1, entity2, weight=weight)

        return G

    def get_insights(self) -> Dict:
        """Generate insights from processed intelligence"""
        if not self.entities:
            return {
                'top_entities': [],
                'most_connected': [],
                'entity_types': {},
                'document_coverage': {}
            }

        # Top entities by frequency
        entity_counts = Counter(ent['text'] for ent in self.entities)
        top_entities = entity_counts.most_common(10)

        # Most connected entities
        G = self.get_entity_network()
        if len(G.nodes()) > 0:
            centrality = nx.degree_centrality(G)
            most_connected = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        else:
            most_connected = []

        # Entity type distribution
        entity_types = Counter(ent['label'] for ent in self.entities)

        # Document coverage
        doc_coverage = Counter(ent['doc_id'] for ent in self.entities)

        return {
            'top_entities': top_entities,
            'most_connected': most_connected,
            'entity_types': dict(entity_types),
            'document_coverage': dict(doc_coverage)
        }

    def export_data(self) -> Dict[str, pd.DataFrame]:
        """Export processed data as DataFrames"""
        entities_df = pd.DataFrame(self.entities) if self.entities else pd.DataFrame()
        relationships_df = pd.DataFrame(self.relationships) if self.relationships else pd.DataFrame()
        reports_df = pd.DataFrame(self.reports) if self.reports else pd.DataFrame()

        return {
            'entities': entities_df,
            'relationships': relationships_df,
            'reports': reports_df
        }

    def clear_data(self):
        """Clear all processed data"""
        self.entities.clear()
        self.relationships.clear()
        self.reports.clear()
        self.document_texts.clear()
        print("🧹 Cleared all processed data")

    def get_entity_timeline(self) -> List[Dict]:
        """Get timeline of entity mentions across documents"""
        timeline = []

        for report in self.reports:
            doc_entities = [ent for ent in self.entities if ent['doc_id'] == report['doc_id']]

            timeline.append({
                'date': report['processed_date'],
                'doc_id': report['doc_id'],
                'doc_type': report['doc_type'],
                'entity_count': len(doc_entities),
                'entities': [ent['text'] for ent in doc_entities]
            })

        return sorted(timeline, key=lambda x: x['date'])

    def search_entities(self, search_term: str, entity_type: str = None) -> List[Dict]:
        """Search for entities by text or type"""
        results = []

        for entity in self.entities:
            text_match = search_term.lower() in entity['text'].lower()
            type_match = entity_type is None or entity['label'] == entity_type

            if text_match and type_match:
                results.append(entity)

        return results

    def get_document_text(self, doc_id: str) -> str:
        """Get original text for a document"""
        return self.document_texts.get(doc_id, "")

    def get_statistics(self) -> Dict:
        """Get processing statistics"""
        return {
            'total_documents': len(self.reports),
            'total_entities': len(self.entities),
            'total_relationships': len(self.relationships),
            'unique_entities': len(set(ent['text'] for ent in self.entities)),
            'entity_types': len(set(ent['label'] for ent in self.entities)),
            'avg_entities_per_doc': len(self.entities) / max(len(self.reports), 1),
            'documents_with_text_stored': len(self.document_texts)
        }


# Testing function
if __name__ == "__main__":
    processor = IntelligenceProcessor()

    sample_text = """INTELLIGENCE REPORT - OPERATION NIGHTFALL
Date: 2025-06-01
Classification: SECRET

Subject: Suspicious Financial Activity - CRIMSON ENTERPRISES

Our financial intelligence unit has identified irregular transactions involving CRIMSON ENTERPRISES, 
a shell company registered in CAYMAN ISLANDS. Analysis shows $15.7 million transferred through 
SWISS NATIONAL BANK to accounts linked to VLADIMIR PETROV.

PETROV, a known associate of the EASTERN SYNDICATE, was previously flagged in Operation THUNDER."""

    entities = processor.process_document(sample_text, "TEST-001", "security_intelligence")
    print(f"\n🧪 Test Results:")
    print(f"📊 Entities extracted: {len(entities)}")
    print(f"📝 Text stored: {bool(processor.get_document_text('TEST-001'))}")
    print(f"📈 Statistics: {processor.get_statistics()}")