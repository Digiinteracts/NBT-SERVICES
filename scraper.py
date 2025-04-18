from flask import Flask, request, jsonify
from flask_cors import CORS
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import requests
import os
import uuid
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import hashlib
import time
import threading

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
# Enable CORS for all routes
# CORS(app, resources={r"/api/*": {"origins": "https://3.110.189.97"}})
CORS(
    app, 
    resources={r"/api/*": {
        "origins": "*",  # Allow all origins for testing
        "methods": ["GET", "POST", "OPTIONS"],  # Include OPTIONS for preflight
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
        "expose_headers": ["Content-Length", "X-Request-Status"],
        "supports_credentials": False,
        "max_age": 3600
    }}
)

# Initialize Pinecone client with your API key
pc = Pinecone(api_key= os.getenv("PINECONE_KEY", "" ))

# Define index name and dimension
INDEX_NAME = "web-scraper-index"
DIMENSION = 512  # Dimension of our vector embeddings

# Global variables for scraping status
scraping_status = {
    "is_scraping": False,
    "total_pages": 0,
    "pages_scraped": 0,
    "urls_found": set(),
    "vectors_processed": 0,
    "current_url": "",
    "error": None
}

# Check if index exists, if not create it (with a free tier compatible configuration)
try:
    # List existing indexes
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if INDEX_NAME not in existing_indexes:
        print(f"Creating new Pinecone index: {INDEX_NAME}")
        # Create index with the correct serverless spec for us-east-1 region
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        # Wait for index to be ready
        time.sleep(10)
    
    # Connect to the index
    index = pc.Index(INDEX_NAME)
    print(f"Successfully connected to index: {INDEX_NAME}")
    use_pinecone = True
    
except Exception as e:
    print(f"Error with Pinecone setup: {e}")
    # Provide a fallback approach - we'll use an in-memory storage for development
    print("Using in-memory storage as fallback")
    in_memory_vectors = []
    use_pinecone = False

def generate_simple_embedding(text, dimension=512):
    """
    Generate a simple deterministic embedding vector for text.
    This is a placeholder - in production, use a proper embedding model.
    """
    # Create a hash of the text
    text_hash = hashlib.md5(text.encode()).hexdigest()
    
    # Convert the hash to a list of numbers
    hash_values = []
    for i in range(0, min(len(text_hash), dimension // 8)):
        hash_values.append(int(text_hash[i:i+2], 16) / 255.0)
    
    # Pad the vector to the required dimension
    embedding = hash_values + [0.0] * (dimension - len(hash_values))
    
    # Normalize the vector
    magnitude = sum(x**2 for x in embedding) ** 0.5
    if magnitude > 0:
        embedding = [x / magnitude for x in embedding]
    
    return embedding

def is_valid_link(link, base_domain):
    """Check if a link is valid for crawling"""
    if not link or link.startswith("mailto:") or link.startswith("javascript:"):
        return False
    
    parsed = urlparse(link)
    base_parsed = urlparse(base_domain)
    
    # Check if the link is in the same domain or subdomain
    return parsed.netloc == "" or parsed.netloc == base_parsed.netloc or parsed.netloc.endswith(f".{base_parsed.netloc}")

def batch_upsert_vectors(vectors):
    """Batch upsert vectors to Pinecone or in-memory storage"""
    if use_pinecone and vectors:
        try:
            # Upsert in batches
            index.upsert(vectors=vectors)
            scraping_status["vectors_processed"] += len(vectors)
        except Exception as e:
            print(f"Error upserting vectors: {e}")
    else:
        # Add to in-memory storage
        global in_memory_vectors
        in_memory_vectors.extend(vectors)
        scraping_status["vectors_processed"] += len(vectors)

def scrape_page(url, base_url, visited, page_limit):
    """Scrape a page and follow links up to the page limit"""
    if url in visited or len(visited) >= page_limit or not scraping_status["is_scraping"]:
        return
    
    print(f"📄 Scraping: {url} ({len(visited) + 1}/{page_limit})")
    visited.add(url)
    scraping_status["urls_found"].add(url)
    scraping_status["current_url"] = url
    scraping_status["pages_scraped"] = len(visited)
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            print(f"❌ Failed to fetch {url}: Status code {response.status_code}")
            return
        
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Try to find the main content div, if not found, use the body
        content_div = soup.find("div", {"id": "s-lg-guide-main"}) or soup.find("main") or soup.body
        
        if content_div:
            elements = content_div.find_all(["h1", "h2", "h3", "p", "li", "a"])
            
            # Prepare batch of vectors
            vectors_batch = []
            
            for el in elements:
                text = el.get_text(strip=True)
                if text and len(text) > 10:  # Only store meaningful content
                    try:
                        # Generate a vector embedding for the text
                        embedding = generate_simple_embedding(text)
                        
                        # Prepare vector for batch insertion
                        record_id = str(uuid.uuid4())
                        metadata = {
                            "source_url": url,
                            "tag": el.name,
                            "text": text
                        }
                        
                        vector = {
                            "id": record_id,
                            "values": embedding,
                            "metadata": metadata
                        }
                        
                        vectors_batch.append(vector)
                        
                        # If batch reaches size limit, upsert
                        if len(vectors_batch) >= 100:
                            batch_upsert_vectors(vectors_batch)
                            vectors_batch = []
                    
                    except Exception as e:
                        print(f"Error processing element: {e}")
            
            # Upsert any remaining vectors
            if vectors_batch:
                batch_upsert_vectors(vectors_batch)
        
        # If we haven't reached the page limit, extract and follow links
        if len(visited) < page_limit and scraping_status["is_scraping"]:
            links = soup.find_all("a", href=True)
            for link in links:
                href = link["href"]
                if is_valid_link(href, base_url):
                    full_url = urljoin(url, href)
                    # Only scrape each URL once
                    if full_url not in visited:
                        scrape_page(full_url, base_url, visited, page_limit)
                    
                    # Stop if we've reached the limit
                    if len(visited) >= page_limit or not scraping_status["is_scraping"]:
                        break
    
    except Exception as e:
        print(f"⚠️ Error scraping {url}: {e}")

def scrape_website_async(start_url, page_limit):
    """Run the scraping process in a separate thread"""
    try:
        # Reset scraping status
        scraping_status["is_scraping"] = True
        scraping_status["total_pages"] = page_limit
        scraping_status["pages_scraped"] = 0
        scraping_status["urls_found"] = set()
        scraping_status["vectors_processed"] = 0
        scraping_status["current_url"] = start_url
        scraping_status["error"] = None
        
        visited = set()
        scrape_page(start_url, start_url, visited, page_limit)
        
        # Mark scraping as complete
        scraping_status["is_scraping"] = False
        print(f"✅ Scraping complete: {len(visited)} pages scraped, {scraping_status['vectors_processed']} vectors processed")
    
    except Exception as e:
        scraping_status["is_scraping"] = False
        scraping_status["error"] = str(e)
        print(f"❌ Scraping error: {e}")

@app.route('/api/scrape', methods=['POST'])
def start_scrape():
    data = request.json
    
    if not data or 'url' not in data:
        return jsonify({"error": "URL is required"}), 400
    
    start_url = data['url']
    page_limit = data.get('limit', 10)  # Default to 100 pages if not specified
    
    # Validate URL
    try:
        parsed_url = urlparse(start_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            return jsonify({"error": "Invalid URL format"}), 400
    except:
        return jsonify({"error": "Invalid URL"}), 400
    
    # Check if scraping is already in progress
    if scraping_status["is_scraping"]:
        return jsonify({
            "status": "in_progress",
            "message": "Scraping already in progress",
            "current_progress": {
                "pages_scraped": scraping_status["pages_scraped"],
                "total_pages": scraping_status["total_pages"],
                "current_url": scraping_status["current_url"],
                "vectors_processed": scraping_status["vectors_processed"]
            }
        }), 200
    
    # Clear previous data if requested
    if data.get('clear_previous', False):
        try:
            if use_pinecone:
                index.delete(delete_all=True)
                print("Deleted all vectors from the index")
            else:
                # Clear in-memory vectors
                global in_memory_vectors
                in_memory_vectors = []
                print("Cleared in-memory vectors")
            
            scraping_status["vectors_processed"] = 0
        except Exception as e:
            print(f"Error deleting vectors: {e}")
    
    # Start scraping in a separate thread
    threading.Thread(target=scrape_website_async, args=(start_url, page_limit), daemon=True).start()
    
    # Return immediate response
    return jsonify({
        "status": "started",
        "message": f"Started scraping {start_url} with a limit of {page_limit} pages"
    }), 200

@app.route('/api/scrape/status', methods=['GET'])
def get_scrape_status():
    """Get the current status of the scraping process"""
    return jsonify({
        "is_scraping": scraping_status["is_scraping"],
        "total_pages": scraping_status["total_pages"],
        "pages_scraped": scraping_status["pages_scraped"],
        "urls_found": list(scraping_status["urls_found"]),
        "vectors_processed": scraping_status["vectors_processed"],
        "current_url": scraping_status["current_url"],
        "error": scraping_status["error"]
    }), 200

@app.route('/api/scrape/cancel', methods=['POST'])
def cancel_scrape():
    """Cancel an in-progress scraping operation"""
    if scraping_status["is_scraping"]:
        scraping_status["is_scraping"] = False
        return jsonify({
            "status": "cancelled",
            "message": "Scraping operation cancelled"
        }), 200
    else:
        return jsonify({
            "status": "not_running",
            "message": "No scraping operation is currently running"
        }), 400

@app.route('/api/search', methods=['GET'])
def search_content():
    query = request.args.get('query', '')
    limit = int(request.args.get('limit', 10))
    
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400
    
    try:
        # Generate embedding for the query
        query_embedding = generate_simple_embedding(query)
        
        results = []
        
        try:
            if use_pinecone:
                search_results = index.query(
                    vector=query_embedding,
                    top_k=limit,
                    include_metadata=True
                )
                
                # Format results
                for match in search_results.matches:
                    results.append({
                        "score": match.score,
                        "source_url": match.metadata.get("source_url"),
                        "tag": match.metadata.get("tag"),
                        "text": match.metadata.get("text")
                    })
            else:
                # Fallback to in-memory search
                # This is a simplified version that calculates cosine similarity
                for vector in in_memory_vectors:
                    # Calculate dot product (simplified cosine similarity)
                    score = sum(a*b for a, b in zip(query_embedding, vector["values"]))
                    results.append({
                        "score": score,
                        "source_url": vector["metadata"]["source_url"],
                        "tag": vector["metadata"]["tag"],
                        "text": vector["metadata"]["text"]
                    })
                
                # Sort by score in descending order and limit results
                results = sorted(results, key=lambda x: x["score"], reverse=True)[:limit]
        
        except Exception as e:
            print(f"Error during search: {e}")
            # Return empty results if search fails
            results = []
        
        return jsonify({
            "query": query,
            "count": len(results),
            "results": results
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    try:
        if use_pinecone:
            # Get index stats
            stats = index.describe_index_stats()
            
            return jsonify({
                "vector_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "storage_type": "pinecone"
            }), 200
        else:
            # Return in-memory stats
            return jsonify({
                "vector_count": len(in_memory_vectors),
                "dimension": DIMENSION,
                "storage_type": "in-memory"
            }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.getenv("PORT", 8080)))