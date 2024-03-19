use anyhow::{Result, Context};
use serde::{Serialize, Deserialize};
use reqwest::{Client, header};
use std::{fs, env, collections::HashMap};

#[derive(Serialize, Deserialize)]
struct Collection {
    name: String,
    vector_size: usize,
    distance: String,
}

#[derive(Serialize, Deserialize)]
struct Point {
    ids: Vec<u64>,
    vectors: Vec<Vec<f32>>,
    payloads: Option<Vec<HashMap<String, serde_json::Value>>>,
}

#[derive(Serialize, Deserialize)]
struct SearchQuery {
    vector: Vec<f32>,
}

#[derive(Serialize, Deserialize)]
struct SearchResult {
    result: Vec<SearchResultItem>,
}

#[derive(Serialize, Deserialize)]
struct SearchResultItem {
    id: u64,
    // Other fields can be included based on the response
}

async fn create_qdrant_collection(client: &Client, collection_name: &str, vector_size: usize, distance: &str) -> Result<()> {
    let collection_body = serde_json::json!({
        "vectors": {
            "size": vector_size,
            "distance": distance
        }
    });

    let response_body = client.put(format!("http://localhost:6333/collections/{}", collection_name))
        .json(&collection_body)
        .send()
        .await?
        .text().await?;

    println!("Response body from creating collection: {}", response_body);
    Ok(())
}

async fn add_points_to_qdrant(client: &Client, collection_name: &str, points: Vec<(u64, Vec<f32>, String)>) -> Result<()> {
    let points_body = serde_json::json!({
        "points": points.iter().map(|(id, vector, chunk)| {
            serde_json::json!({
                "id": id,
                "vector": vector,
                "payload": {"text": chunk}
            })
        }).collect::<Vec<_>>()
    });

    let response_body = client.put(format!("http://localhost:6333/collections/{}/points", collection_name))
        .json(&points_body)
        .send()
        .await?
        .text().await?;

    println!("Response body from adding points: {}", response_body);
    Ok(())
}

async fn embed_text(client: &Client, text: String) -> Result<Vec<f32>> {
    let api_key = env::var("HUGGINGFACE_API_KEY").context("Expected a Hugging Face API key")?;
    let mut headers = header::HeaderMap::new();
    headers.insert(
        "Authorization",
        header::HeaderValue::from_str(&format!("Bearer {}", api_key))?
    );

    let request_payload = serde_json::json!([text]);

    let response = client.post("https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2")
        .headers(headers)
        .json(&request_payload)
        .send()
        .await?
        .text().await?;

    println!("Raw API response: {}", response);

    // Deserialize the response into a vector of f32. Adjust this according to the actual response structure.
    let embedding: Vec<f32> = serde_json::from_str(&response)
        .context("Failed to deserialize embedding response")?;

    Ok(embedding)
}

async fn load_data_to_qdrant(client: &Client, texts: Vec<String>, collection_name: &str, vector_size: usize, distance: &str) -> Result<()> {
    println!("Creating new collection in Qdrant...");
    create_qdrant_collection(client, collection_name, vector_size, distance).await?;

    let mut points: Vec<(u64, Vec<f32>, String)> = vec![];

    for (i, text) in texts.iter().enumerate() {
        let vector = embed_text(client, text.clone()).await?;
        points.push((i as u64, vector, text.clone())); // Assuming text itself as payload, modify as needed.
    }

    println!("Inserting new data into Qdrant...");
    add_points_to_qdrant(client, collection_name, points).await?;

    println!("Finished inserting data into Qdrant.");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let client = reqwest::Client::new();
    let args: Vec<String> = env::args().collect();

    match args.get(1).map(String::as_str) {
        Some("load") => {
            let file_content = fs::read_to_string("src/reg-all.txt").context("Failed to read from reg-all.txt")?;
            let texts: Vec<String> = file_content.split("\n\n").map(String::from).collect();
            load_data_to_qdrant(&client, texts, "registration_collection", 384, "Cosine").await?;
            println!("Data loaded successfully");
        }
        Some(query) => {
            let query_vector = embed_text(&client, query.to_string()).await?;
            let search_query = SearchQuery { vector: query_vector };
            let search_result: SearchResult = client.post(format!("http://localhost:6333/collections/registration_collection/points/search"))
                .json(&search_query)
                .send()
                .await?
                .json()
                .await?;

            let most_similar_id = search_result.result.get(0)
                .context("No similar vector found")?.id;

            println!("Most similar text ID: {}", most_similar_id);
        }
        None => println!("No valid arguments provided"),
    }

    Ok(())
}
