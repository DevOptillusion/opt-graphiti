#!/usr/bin/env python3
"""
Export nodes from Neo4j database to CSV format
This script connects to your Neo4j database and exports all nodes to CSV files
"""

import os
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

# Neo4j connection parameters
NEO4J_URI = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.environ.get('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD', 'Echo0228')

def export_nodes_to_csv(output_dir="neo4j-export"):
    """Export all nodes from Neo4j to CSV files"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Connect to Neo4j
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            # Get all node labels
            labels_query = "CALL db.labels()"
            labels_result = session.run(labels_query)
            labels = [record["label"] for record in labels_result]
            
            print(f"Found node labels: {labels}")
            
            # Export nodes for each label
            for label in labels:
                print(f"Exporting {label} nodes...")
                
                # Get all properties for this label
                properties_query = f"""
                MATCH (n:{label})
                WITH n
                UNWIND keys(n) as key
                RETURN DISTINCT key
                """
                properties_result = session.run(properties_query)
                properties = [record["key"] for record in properties_result]
                
                if not properties:
                    print(f"No properties found for {label}")
                    continue
                
                # Export nodes with all properties
                export_query = f"""
                MATCH (n:{label})
                RETURN n
                """
                nodes_result = session.run(export_query)
                
                # Convert to list of dictionaries
                nodes_data = []
                for record in nodes_result:
                    node = dict(record["n"])
                    # Add label information
                    node["~labels"] = label
                    nodes_data.append(node)
                
                if nodes_data:
                    # Create DataFrame and save to CSV
                    df = pd.DataFrame(nodes_data)
                    csv_path = os.path.join(output_dir, f"{label.lower()}-export.csv")
                    df.to_csv(csv_path, index=False, encoding='utf-8')
                    print(f"Exported {len(nodes_data)} {label} nodes to {csv_path}")
                else:
                    print(f"No {label} nodes found")
            
            # Also export all nodes in a single file (like Bloom export format)
            print("Exporting all nodes to single file...")
            all_nodes_query = """
            MATCH (n)
            RETURN n, labels(n) as node_labels
            """
            all_nodes_result = session.run(all_nodes_query)
            
            all_nodes_data = []
            for record in all_nodes_result:
                node = dict(record["n"])
                node["~labels"] = ":".join(record["node_labels"])
                all_nodes_data.append(node)
            
            if all_nodes_data:
                df_all = pd.DataFrame(all_nodes_data)
                csv_path_all = os.path.join(output_dir, "node-export.csv")
                df_all.to_csv(csv_path_all, index=False, encoding='utf-8')
                print(f"Exported {len(all_nodes_data)} total nodes to {csv_path_all}")
            
            # Export relationships
            print("Exporting relationships...")
            relationships_query = """
            MATCH (a)-[r]->(b)
            RETURN 
                id(a) as start_id,
                id(b) as end_id,
                type(r) as rel_type,
                r as properties
            """
            relationships_result = session.run(relationships_query)
            
            relationships_data = []
            for record in relationships_result:
                rel_data = {
                    "~start": record["start_id"],
                    "~end": record["end_id"],
                    "~type": record["rel_type"]
                }
                # Add relationship properties
                rel_props = dict(record["properties"])
                rel_data.update(rel_props)
                relationships_data.append(rel_data)
            
            if relationships_data:
                df_rel = pd.DataFrame(relationships_data)
                csv_path_rel = os.path.join(output_dir, "relationship-export.csv")
                df_rel.to_csv(csv_path_rel, index=False, encoding='utf-8')
                print(f"Exported {len(relationships_data)} relationships to {csv_path_rel}")
            
            # Export graph summary
            print("Exporting graph summary...")
            summary_query = """
            MATCH (n)
            RETURN 
                labels(n) as labels,
                count(n) as count
            """
            summary_result = session.run(summary_query)
            
            summary_data = []
            for record in summary_result:
                summary_data.append({
                    "labels": ":".join(record["labels"]),
                    "count": record["count"]
                })
            
            if summary_data:
                df_summary = pd.DataFrame(summary_data)
                csv_path_summary = os.path.join(output_dir, "graph-summary.csv")
                df_summary.to_csv(csv_path_summary, index=False, encoding='utf-8')
                print(f"Exported graph summary to {csv_path_summary}")
    
    finally:
        driver.close()
        print("Export completed!")

if __name__ == "__main__":
    import sys
    
    # Get output directory from command line argument or use default
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "neo4j-export"
    
    print(f"Starting Neo4j node export to {output_dir}...")
    export_nodes_to_csv(output_dir)
