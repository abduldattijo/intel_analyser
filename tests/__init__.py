import spacy
import pandas as pd
from collections import defaultdict, Counter
import networkx as nx
from datetime import datetime
import re
import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)


class IntelligenceProcessor:
    def __init__(self):
        try:
            # Load spaCy model with error handling
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy model loaded successfully")
        except OSError:
            print("❌ spaCy model not found. Run: python -m spacy download en_core_web_sm")
            raise

        self.reports = []
        self.entities = []
        self.relationships = []

    def process_document(self, text, doc_id, doc_type="intelligence_report"):
        """Process a single document and extract entities and relationships"""
        if not text or not text.strip():
            print(f"⚠️  Warning: Empty document {doc_id}")
            return []

        print(f"🔄 Processing document: {doc_id}")
        doc = self.nlp(text)

        # Extract entities with improved filtering
        doc_entities = []
        for ent in doc.ents:
            # Filter out very short entities and common stopwords
            if (ent.label_ in ["PERSON", "ORG", "GPE", "EVENT", "MONEY", "DATE", "PRODUCT"]
                    and len(ent.text.strip()) > 2
                    and not ent.text.lower() in ['the', 'and', 'or', 'but', 'in', 'on', 'at']):
                entity_data = {
                    'text': ent.text.strip(),
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'doc_id': doc_id,
                    'doc_type': doc_type,
                    'confidence': ent.label_ if hasattr(ent, 'label_') else 'UNKNOWN'
                }
                doc_entities.append(entity_data)
                self.entities.append(entity_data)

        # Extract relationships (improved co-occurrence with distance weighting)
        entities_in_doc = [ent['text'] for ent in doc_entities]
        for i, ent1 in enumerate(entities_in_doc):
            for j, ent2 in enumerate(entities_in_doc[i + 1:], i + 1):
                # Calculate sentence distance for relationship strength
                sentence_distance = abs(doc_entities[i]['start'] - doc_entities[j]['start']) / len(text)

                self.relationships.append({
                    'entity1': ent1,
                    'entity2': ent2,
                    'doc_id': doc_id,
                    'relationship_type': 'co_occurrence',
                    'strength': max(0.1, 1.0 - sentence_distance)  # Closer entities = stronger relationship
                })

        # Store document metadata
        self.reports.append({
            'doc_id': doc_id,
            'doc_type': doc_type,
            'entity_count': len(doc_entities),
            'processed_date': datetime.now(),
            'text_length': len(text),
            'word_count': len(text.split())
        })

        print(f"✅ Extracted {len(doc_entities)} entities from {doc_id}")
        return doc_entities

    def get_entity_network(self, min_weight=0.5):
        """Create network graph of entity relationships with improved weighting"""
        G = nx.Graph()

        # Add entities as nodes with metadata
        entity_counts = Counter([ent['text'] for ent in self.entities])
        entity_types = {}
        for ent in self.entities:
            entity_types[ent['text']] = ent['label']

        for entity, count in entity_counts.items():
            G.add_node(entity,
                       frequency=count,
                       entity_type=entity_types.get(entity, 'UNKNOWN'),
                       size=min(50, max(10, count * 5)))

        # Add relationships as weighted edges
        relationship_weights = defaultdict(float)
        for rel in self.relationships:
            key = tuple(sorted([rel['entity1'], rel['entity2']]))
            relationship_weights[key] += rel.get('strength', 1.0)

        for (ent1, ent2), weight in relationship_weights.items():
            if weight >= min_weight and ent1 in G.nodes() and ent2 in G.nodes():
                G.add_edge(ent1, ent2, weight=weight,
                           edge_type='co_occurrence',
                           strength=weight)

        print(f"📊 Network created: {len(G.nodes())} nodes, {len(G.edges())} edges")
        return G

    def get_insights(self):
        """Generate basic insights from processed data"""
        insights = {}

        # Most frequent entities
        entity_counts = Counter([ent['text'] for ent in self.entities])
        insights['top_entities'] = entity_counts.most_common(10)

        # Entity types distribution
        type_counts = Counter([ent['label'] for ent in self.entities])
        insights['entity_types'] = dict(type_counts)

        # Document types processed
        doc_type_counts = Counter([report['doc_type'] for report in self.reports])
        insights['document_types'] = dict(doc_type_counts)

        # Most connected entities (high relationship count)
        relationship_counts = defaultdict(int)
        for rel in self.relationships:
            relationship_counts[rel['entity1']] += 1
            relationship_counts[rel['entity2']] += 1

        insights['most_connected'] = sorted(relationship_counts.items(),
                                            key=lambda x: x[1], reverse=True)[:10]

        return insights

    def export_data(self):
        """Export processed data for analysis"""
        return {
            'entities': pd.DataFrame(self.entities),
            'relationships': pd.DataFrame(self.relationships),
            'reports': pd.DataFrame(self.reports)
        }


# Example usage
if __name__ == "__main__":
    processor = IntelligenceProcessor()

    # Sample intelligence report
    sample_report = """
    INTELLIGENCE REPORT - OPERATION THUNDERBOLT
    Date: 2025-06-08
    Classification: CONFIDENTIAL

    Subject: Suspicious Activity - Port of Lagos

    Our surveillance team observed unusual shipping activity at the Port of Lagos 
    involving the vessel MV ATLANTIC STAR. The ship, registered to GLOBAL SHIPPING LTD, 
    arrived from ROTTERDAM on June 5th. Intelligence suggests potential involvement 
    of VICTOR MARTINEZ, a known associate of the CRIMSON SYNDICATE.

    Financial analysis indicates transactions totaling $2.5 million through 
    OFFSHORE BANK OF CYPRUS. Recommend continued surveillance and coordination 
    with CUSTOMS AUTHORITY.

    Agent: SARAH JOHNSON
    Next Review: June 15, 2025
    """

    # Process the sample report
    entities = processor.process_document(sample_report, "RPT-001", "security_intelligence")

    # Get insights
    insights = processor.get_insights()
    print("Extracted Entities:")
    for ent in entities:
        print(f"- {ent['text']} ({ent['label']})")

    print("\nInsights:")
    print(f"Total entities: {len(processor.entities)}")
    print(f"Entity types: {insights['entity_types']}")