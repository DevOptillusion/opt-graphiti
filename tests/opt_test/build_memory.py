import asyncio
import json
import logging
import os

from datetime import datetime, timezone
from logging import INFO

from dotenv import load_dotenv

from graphiti_core.llm_client import LLMConfig, GeminiClient

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config import SearchConfig
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF
from pytz import timezone as pytz_timezone

# Import graph schema
from graph_schema import entity_types, edge_types, edge_type_map

# Configure logging
logging.basicConfig(
    level=INFO,  # Changed from INFO to DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
load_dotenv(dotenv_path=env_file)

def get_llm_client():
    """Get LLM client with fallback options"""
    gemini_api_key = os.environ.get('GEMINI_API_KEY_MEMORY')
    
    if not gemini_api_key:
        raise ValueError(
            'GEMINI_API_KEY_MEMORY environment variable is not set.\n'
            'Please set it in one of the following ways:\n'
            '1. Create a .env file in the project root with: GEMINI_API_KEY_MEMORY=your_key\n'
            '2. Set it in PyCharm: Run -> Edit Configurations -> Environment Variables\n'
            '3. Export it in terminal: export GEMINI_API_KEY_MEMORY=your_key'
        )
    
    gemini_config = LLMConfig(
        api_key=gemini_api_key,
        model="gemini-2.5-flash",
        small_model="gemini-2.5-flash",
        max_tokens=8192
    )
    return GeminiClient(config=gemini_config)

def get_neo4j_config():
    """Get Neo4j configuration"""
    neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
    neo4j_password = os.environ.get('NEO4J_PASSWORD', 'Echo0228')
    
    if not neo4j_uri or not neo4j_user or not neo4j_password:
        raise ValueError('NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set')
    
    return neo4j_uri, neo4j_user, neo4j_password

async def create_custom_relationship_types(graphiti):
    """Create custom relationship types based on the name property of RELATES_TO relationships"""
    try:
        # Get all RELATES_TO relationships with their name properties
        query = """
        MATCH (source)-[r:RELATES_TO]->(target)
        WHERE r.name IS NOT NULL
        RETURN source.uuid AS source_uuid, target.uuid AS target_uuid, r.name AS relationship_name, r.uuid AS relationship_uuid, r.fact AS fact, r.group_id AS group_id, r.created_at AS created_at, r.episodes AS episodes, r.expired_at AS expired_at, r.valid_at AS valid_at, r.invalid_at AS invalid_at, r.fact_embedding AS fact_embedding
        """
        
        records, _, _ = await graphiti.driver.execute_query(query)
        
        for record in records:
            relationship_name = record['relationship_name']
            source_uuid = record['source_uuid']
            target_uuid = record['target_uuid']
            relationship_uuid = record['relationship_uuid']
            
            # First, create the new relationship (with duplicate check)
            create_query = f"""
            MATCH (source {{uuid: $source_uuid}})
            MATCH (target {{uuid: $target_uuid}})
            WHERE NOT EXISTS((source)-[:{relationship_name} {{uuid: $relationship_uuid}}]->(target))
            CREATE (source)-[new:{relationship_name} {{
                uuid: $relationship_uuid,
                name: $relationship_name,
                fact: $fact,
                fact_embedding: $fact_embedding,
                group_id: $group_id,
                created_at: $created_at,
                episodes: $episodes,
                expired_at: $expired_at,
                valid_at: $valid_at,
                invalid_at: $invalid_at
            }}]->(target)
            """
            
            # Then, delete the old relationship
            delete_query = """
            MATCH (source {uuid: $source_uuid})-[old:RELATES_TO {uuid: $relationship_uuid}]->(target {uuid: $target_uuid})
            DELETE old
            RETURN count(old) as deleted_count
            """
            
            # Execute create query
            create_result, _, _ = await graphiti.driver.execute_query(
                create_query,
                relationship_name=relationship_name,
                source_uuid=source_uuid,
                target_uuid=target_uuid,
                relationship_uuid=relationship_uuid,
                fact=record['fact'],
                fact_embedding=record['fact_embedding'],
                group_id=record['group_id'],
                created_at=record['created_at'],
                episodes=record['episodes'],
                expired_at=record['expired_at'],
                valid_at=record['valid_at'],
                invalid_at=record['invalid_at']
            )
            
            # Always try to delete the old relationship, regardless of create result
            # (since the WHERE clause in create_query handles duplicates)
            delete_result, _, _ = await graphiti.driver.execute_query(
                delete_query,
                source_uuid=source_uuid,
                target_uuid=target_uuid,
                relationship_uuid=relationship_uuid
            )
            
            print(f"Created relationship {relationship_name} between {source_uuid} and {target_uuid}")
            print(f"Delete result: {delete_result}")
            
        print(f"Created {len(records)} custom relationship types")
        
        # Check if any RELATES_TO relationships remain
        check_query = """
        MATCH ()-[r:RELATES_TO]->()
        RETURN count(r) as remaining_relates_to
        """
        remaining_result, _, _ = await graphiti.driver.execute_query(check_query)
        if remaining_result:
            remaining_count = remaining_result[0]['remaining_relates_to']
            print(f"Warning: {remaining_count} RELATES_TO relationships still exist")
        
    except Exception as e:
        print(f"Error creating custom relationship types: {e}")

async def clear_database(graphiti):
    """Clear all nodes and relationships from the database"""
    print("🗑️  Clearing database...")
    try:
        # Delete all relationships first
        await graphiti.driver.execute_query("MATCH ()-[r]->() DELETE r")
        print("   Deleted all relationships")
        
        # Delete all nodes
        await graphiti.driver.execute_query("MATCH (n) DELETE n")
        print("   Deleted all nodes")
        
        print("✅ Database cleared successfully!")
    except Exception as e:
        print(f"❌ Error clearing database: {e}")

async def export_database(memory_build_acts=None):
    """Export the database after processing"""
    print("📤 Exporting database...")
    try:
        import subprocess
        import sys
        import os
        import json
        from datetime import datetime
        
        # Create main neo4j_data directory
        main_dir = "neo4j_data"
        os.makedirs(main_dir, exist_ok=True)
        
        # Create timestamped subdirectory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(main_dir, timestamp)
        
        # Export database using existing script with the same Python interpreter
        result = subprocess.run([sys.executable, 'export_nodes.py', output_dir], 
                              capture_output=True, text=True, cwd='.')
        if result.returncode == 0:
            print(f"✅ Database exported successfully to {output_dir}/")
        else:
            print(f"❌ Export failed: {result.stderr}")
        
        # Export memory_build_acts if provided
        if memory_build_acts is not None:
            memory_output_file = os.path.join(output_dir, "memory_build_acts.json")
            with open(memory_output_file, 'w', encoding='utf-8') as f:
                json.dump(memory_build_acts, f, indent=2, ensure_ascii=False)
            print(f"✅ Memory build acts exported to {memory_output_file}")
        
        # Return the output directory path for reuse
        return output_dir
            
    except Exception as e:
        print(f"❌ Error during export: {e}")
        return None

async def initialize_database(graphiti, clear_db=False):
    """
    Initialize the database with indices and constraints.
    This only needs to be done once when setting up the database.
    
    Args:
        graphiti: Graphiti instance
        clear_db: Whether to clear the database before initializing (default: False)
    """
    # Clear the database before starting (optional)
    if clear_db:
        await clear_database(graphiti)
    
    # Initialize the graph database with graphiti's indices. This only needs to be done once.
    await graphiti.build_indices_and_constraints()
    
    # Create custom fulltext index for your node types
    print("Creating custom fulltext index for custom node types...")
    try:
        # Create fulltext index for person_search
        custom_index_query = """CREATE FULLTEXT INDEX person_search IF NOT EXISTS
        FOR (n:Person) 
        ON EACH [n.name, n.summary]"""
        
        await graphiti.driver.execute_query(custom_index_query)
        print("Custom fulltext index 'person_search' created successfully!")

        # Create fulltext index for relationship_view_search
        custom_index_query = """CREATE FULLTEXT INDEX relationship_view_search IF NOT EXISTS
        FOR (n:RelationshipView) 
        ON EACH [n.name, n.summary]"""
        
        await graphiti.driver.execute_query(custom_index_query)
        print("Custom fulltext index 'relationship_view_search' created successfully!")

        # Create fulltext index for preference_search
        custom_index_query = """CREATE FULLTEXT INDEX preference_search IF NOT EXISTS
        FOR (n:Preference) 
        ON EACH [n.name, n.summary]"""
        
        await graphiti.driver.execute_query(custom_index_query)
        print("Custom fulltext index 'preference_search' created successfully!")
        
        # Create fulltext index for belief_search
        custom_index_query = """CREATE FULLTEXT INDEX belief_search IF NOT EXISTS
        FOR (n:Belief) 
        ON EACH [n.name, n.summary]"""
        
        await graphiti.driver.execute_query(custom_index_query)
        print("Custom fulltext index 'belief_search' created successfully!")
        
        # Create fulltext index for memory_note_search
        custom_index_query = """CREATE FULLTEXT INDEX memory_note_search IF NOT EXISTS
        FOR (n:MemoryNote) 
        ON EACH [n.name, n.summary]"""
        
        await graphiti.driver.execute_query(custom_index_query)
        print("Custom fulltext index 'memory_note_search' created successfully!")
        
    except Exception as e:
        print(f"Error creating custom index: {e}")

async def process_episodes_data(graphiti, memory_build_episodes):
    """
    Process episode data and add to the database.
    
    Args:
        graphiti: Graphiti instance
        memory_build_episodes: List of episode data to process
    """
    for i, episode in enumerate(memory_build_episodes):
        group_id = f'{episode["act_id"]}_{episode["episode_id"]}'
        
        # Check if episode with this group_id already exists
        check_query = """
        MATCH (e:Episodic {group_id: $group_id})
        RETURN count(e) as episode_count
        """
        result, _, _ = await graphiti.driver.execute_query(check_query, group_id=group_id)
        
        if result and result[0]['episode_count'] > 0:
            print(f"Episode with group_id '{group_id}' already exists, skipping...")
            continue
        
        await graphiti.add_episode(
            name=f'act_{episode["act_id"]}_episode_{episode["episode_id"]}',
            group_id=group_id,
            episode_body=episode['content'] if isinstance(episode['content'], str) else json.dumps(episode['content']),
            source=EpisodeType.text,
            source_description='act_input_metadata',
            reference_time=datetime.now(pytz_timezone('US/Pacific')),
            excluded_entity_types=["Entity"],  # Exclude the default Entity type
            entity_types=entity_types,
            edge_types=edge_types,
            edge_type_map=edge_type_map
        )

async def process_acts(file_path=None, memory_build_episodes=None):
    """
    Wrapper function to process episodes that can be called from prepare_input.py
    
    Args:
        file_path: Path to the JSON file containing episodes (optional if memory_build_episodes provided)
        memory_build_episodes: List of episode data to process directly, or None to load from file
    
    Returns:
        Result of the processing
    """
    llm_client = get_llm_client()
    neo4j_uri, neo4j_user, neo4j_password = get_neo4j_config()
    
    graphiti = Graphiti(
        neo4j_uri, neo4j_user, neo4j_password,
        llm_client=llm_client
    )
    try:
        # Initialize database (only needed once)
        # clear_db=True will clear all existing data in the database
        await initialize_database(graphiti, clear_db=True)
        
        # Debug: Check what indexes and procedures are available
        print("=== Database Debug Info ===")
        # try:
        #     # Check all indexes
        #     indexes_query = "SHOW INDEXES YIELD name, type, labelsOrTypes, properties"
        #     indexes, _, _ = await graphiti.driver.execute_query(indexes_query)
        #     #print(f"All indexes: {indexes}")
            
        #     # Check fulltext indexes specifically
        #     fulltext_query = "SHOW INDEXES YIELD name, type WHERE type = 'FULLTEXT'"
        #     fulltext_indexes, _, _ = await graphiti.driver.execute_query(fulltext_query)
        #     print(f"Fulltext indexes: {fulltext_indexes}")
            
        #     # Check available procedures
        #     procedures_query = "SHOW PROCEDURES YIELD name WHERE name CONTAINS 'fulltext' OR name CONTAINS 'index'"
        #     procedures, _, _ = await graphiti.driver.execute_query(procedures_query)
        #     print(f"Relevant procedures: {procedures}")
            
        # except Exception as e:
        #     print(f"Error checking database: {e}")
        # print("=== End Debug Info ===")
        
        # Handle episode data - either provided directly or load from file
        if memory_build_episodes is None:
            # Load episodes from JSON file if no episodes provided
            if file_path is None:
                file_path = 'real_act_samples.json'
            with open(file_path, 'r', encoding='utf-8') as f:
                episodes = json.load(f)
            # WORKAROUND: Due to a bug in graphiti_core with custom entity types,
            # we can only process one episode at a time
            memory_build_episodes = episodes[:1]  # Process only first episode
            print(f"⚠️  Note: Processing only 1 episode at a time due to graphiti_core bug with custom entity types")
        
        # Process the episode data
        await process_episodes_data(graphiti, memory_build_episodes)
        
        # Create custom relationship types after processing all episodes -retired as usuing APOC merge relationship
        # await create_custom_relationship_types(graphiti)
        
        print("🎉 Processing completed successfully!")
        return {"status": "success", "episodes_processed": len(memory_build_episodes)}

    finally:
        # Close the connection
        await graphiti.close()
        print('\nConnection closed')
        
        # Export the database after processing
        await export_database()

async def main():
    """Original main function - calls the wrapper with default parameters"""
    return await process_acts(file_path='real_act_samples.json')

if __name__ == '__main__':
    asyncio.run(main())
