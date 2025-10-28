from dotenv import load_dotenv
load_dotenv()

import os
from neo4j import GraphDatabase

# Test Neo4j connection
uri = os.getenv('NEO4J_URI')
username = os.getenv('NEO4J_USERNAME')
password = os.getenv('NEO4J_PASSWORD')

print(f"Connecting to: {uri}")
print(f"Username: {username}")

driver = GraphDatabase.driver(uri, auth=(username, password))

try:
    # Verify connectivity
    driver.verify_connectivity()
    print("✅ Neo4j connection successful!")
    
    # Count nodes
    with driver.session() as session:
        result = session.run("MATCH (n) RETURN count(n) as count")
        count = result.single()["count"]
        print(f"✅ Total nodes in database: {count}")
        
        # Get node labels
        result = session.run("CALL db.labels()")
        labels = [record["label"] for record in result]
        print(f"✅ Node labels: {', '.join(labels) if labels else 'None'}")
        
except Exception as e:
    print(f"❌ Error: {e}")
finally:
    driver.close()
    print("Connection closed")
