use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Encoding method for perception vectors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EncodingMethod {
    Raw,
    Normalized,
    HashProjection,
    RandomProjection,
    LearnedProjection,
}

/// A single sensor reading.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensorReading {
    pub sensor_type: String,
    pub value: f64,
    pub unit: String,
    pub timestamp: u64,
    pub confidence: f64,
}

impl SensorReading {
    pub fn new(sensor_type: &str, value: f64, unit: &str) -> Self {
        Self {
            sensor_type: sensor_type.to_string(),
            value,
            unit: unit.to_string(),
            timestamp: 0,
            confidence: 1.0,
        }
    }

    /// Convert to raw vector: [value, confidence, timestamp_norm]
    pub fn to_raw_vector(&self) -> Vec<f64> {
        // Normalize timestamp to [0, 1] range using simple modular approach
        let timestamp_norm = (self.timestamp as f64 % 1_000_000.0) / 1_000_000.0;
        vec![self.value, self.confidence, timestamp_norm]
    }
}

/// Perception vector produced by encoding sensor readings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerceptionVector {
    pub id: Uuid,
    pub vector: Vec<f64>,
    pub source_readings: Vec<SensorReading>,
    pub room_id: String,
    pub timestamp: u64,
}

impl PerceptionVector {
    pub fn cosine_similarity(a: &Self, b: &Self) -> f64 {
        let dot: f64 = a.vector.iter().zip(&b.vector).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.vector.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.vector.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        dot / (norm_a * norm_b)
    }

    pub fn euclidean(a: &Self, b: &Self) -> f64 {
        a.vector
            .iter()
            .zip(&b.vector)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

/// Statistics about a perception batch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerceptionStats {
    pub count: usize,
    pub avg_norm: f64,
    pub dimension: usize,
}

/// A batch of perception vectors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerceptionBatch {
    pub vectors: Vec<PerceptionVector>,
    pub encoding_method: EncodingMethod,
}

impl PerceptionBatch {
    /// Find k nearest neighbors to query by euclidean distance.
    pub fn nearest(&self, query: &PerceptionVector, k: usize) -> Vec<&PerceptionVector> {
        let mut indexed: Vec<(f64, usize)> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (PerceptionVector::euclidean(query, v), i))
            .collect();
        indexed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        indexed.into_iter().take(k).map(|(_, i)| &self.vectors[i]).collect()
    }

    pub fn stats(&self) -> PerceptionStats {
        let count = self.vectors.len();
        if count == 0 {
            return PerceptionStats {
                count: 0,
                avg_norm: 0.0,
                dimension: 0,
            };
        }
        let dimension = self.vectors[0].vector.len();
        let avg_norm: f64 = self
            .vectors
            .iter()
            .map(|v| v.vector.iter().map(|x| x * x).sum::<f64>().sqrt())
            .sum::<f64>()
            / count as f64;
        PerceptionStats {
            count,
            avg_norm,
            dimension,
        }
    }
}

/// Encoder that converts sensor readings into perception vectors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerceptionEncoder {
    pub input_dim: usize,
    pub output_dim: usize,
    pub method: EncodingMethod,
}

impl PerceptionEncoder {
    pub fn new(input_dim: usize, output_dim: usize, method: EncodingMethod) -> Self {
        Self {
            input_dim,
            output_dim,
            method,
        }
    }

    pub fn encode(&self, readings: &[SensorReading]) -> PerceptionVector {
        let vector = match &self.method {
            EncodingMethod::Raw => self.encode_raw(readings),
            EncodingMethod::Normalized => self.encode_normalized(readings),
            EncodingMethod::HashProjection => self.encode_hash(readings),
            EncodingMethod::RandomProjection => self.encode_random(readings),
            EncodingMethod::LearnedProjection => self.encode_learned(readings),
        };
        let ts = readings.iter().map(|r| r.timestamp).max().unwrap_or(0);
        PerceptionVector {
            id: Uuid::new_v4(),
            vector,
            source_readings: readings.to_vec(),
            room_id: String::new(),
            timestamp: ts,
        }
    }

    pub fn encode_batch(&self, readings_batch: &[Vec<SensorReading>]) -> PerceptionBatch {
        let vectors = readings_batch.iter().map(|r| self.encode(r)).collect();
        PerceptionBatch {
            vectors,
            encoding_method: self.method.clone(),
        }
    }

    fn encode_raw(&self, readings: &[SensorReading]) -> Vec<f64> {
        let raw: Vec<f64> = readings.iter().flat_map(|r| r.to_raw_vector()).collect();
        self.resize(raw)
    }

    fn encode_normalized(&self, readings: &[SensorReading]) -> Vec<f64> {
        let normalized = normalize_readings(readings);
        let raw: Vec<f64> = normalized.iter().flat_map(|r| r.to_raw_vector()).collect();
        self.resize(raw)
    }

    fn encode_hash(&self, readings: &[SensorReading]) -> Vec<f64> {
        let raw: Vec<f64> = readings.iter().flat_map(|r| r.to_raw_vector()).collect();
        let projected = self.deterministic_project(&raw, "hash");
        self.resize(projected)
    }

    fn encode_random(&self, readings: &[SensorReading]) -> Vec<f64> {
        let raw: Vec<f64> = readings.iter().flat_map(|r| r.to_raw_vector()).collect();
        let projected = self.deterministic_project(&raw, "random");
        self.resize(projected)
    }

    fn encode_learned(&self, readings: &[SensorReading]) -> Vec<f64> {
        let raw: Vec<f64> = readings.iter().flat_map(|r| r.to_raw_vector()).collect();
        let projected = self.deterministic_project(&raw, "learned");
        self.resize(projected)
    }

    /// Deterministic projection using a simple hash-based seed for reproducibility.
    fn deterministic_project(&self, input: &[f64], seed: &str) -> Vec<f64> {
        let mut output = vec![0.0; self.output_dim];
        for (i, &val) in input.iter().enumerate() {
            // Simple deterministic mixing: use position and seed
            let hash = format!("{}-{}-{}", seed, i, self.output_dim);
            let h = hash_bytes(&hash);
            for j in 0..self.output_dim {
                let mix = ((h.wrapping_add((j as u64).wrapping_mul(2654435761))) % 1000000) as f64
                    / 1000000.0
                    - 0.5;
                output[j] += val * mix;
            }
        }
        let norm: f64 = output.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            for x in output.iter_mut() {
                *x /= norm;
            }
        }
        output
    }

    fn resize(&self, mut v: Vec<f64>) -> Vec<f64> {
        v.resize(self.output_dim, 0.0);
        v
    }
}

/// Simple hash function for deterministic projections.
fn hash_bytes(s: &str) -> u64 {
    let mut h: u64 = 14695981039346656037;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
}

/// Z-score normalize the values of a set of sensor readings.
pub fn normalize_readings(readings: &[SensorReading]) -> Vec<SensorReading> {
    if readings.is_empty() {
        return vec![];
    }
    let n = readings.len() as f64;
    let mean: f64 = readings.iter().map(|r| r.value).sum::<f64>() / n;
    let variance: f64 = readings.iter().map(|r| (r.value - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();
    if std_dev == 0.0 {
        return readings.to_vec();
    }
    readings
        .iter()
        .map(|r| {
            let mut normalized = r.clone();
            normalized.value = (r.value - mean) / std_dev;
            normalized
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sensor_reading_creation() {
        let r = SensorReading::new("temperature", 23.5, "celsius");
        assert_eq!(r.sensor_type, "temperature");
        assert_eq!(r.value, 23.5);
        assert_eq!(r.unit, "celsius");
        assert_eq!(r.confidence, 1.0);
        assert_eq!(r.timestamp, 0);
    }

    #[test]
    fn test_sensor_reading_raw_vector() {
        let r = SensorReading {
            sensor_type: "temp".into(),
            value: 10.0,
            unit: "C".into(),
            timestamp: 500_000,
            confidence: 0.8,
        };
        let v = r.to_raw_vector();
        assert_eq!(v.len(), 3);
        assert_eq!(v[0], 10.0);
        assert_eq!(v[1], 0.8);
        assert!((v[2] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_encode_raw() {
        let enc = PerceptionEncoder::new(3, 6, EncodingMethod::Raw);
        let readings = vec![SensorReading::new("temp", 20.0, "C")];
        let pv = enc.encode(&readings);
        assert_eq!(pv.vector.len(), 6);
        assert_eq!(pv.source_readings.len(), 1);
    }

    #[test]
    fn test_encode_normalized() {
        let enc = PerceptionEncoder::new(3, 6, EncodingMethod::Normalized);
        let readings = vec![
            SensorReading::new("temp", 10.0, "C"),
            SensorReading::new("temp", 20.0, "C"),
        ];
        let pv = enc.encode(&readings);
        assert_eq!(pv.vector.len(), 6);
    }

    #[test]
    fn test_encode_hash_projection() {
        let enc = PerceptionEncoder::new(3, 4, EncodingMethod::HashProjection);
        let readings = vec![SensorReading::new("temp", 5.0, "C")];
        let pv = enc.encode(&readings);
        assert_eq!(pv.vector.len(), 4);
    }

    #[test]
    fn test_encode_random_projection() {
        let enc = PerceptionEncoder::new(3, 8, EncodingMethod::RandomProjection);
        let readings = vec![SensorReading::new("temp", 5.0, "C")];
        let pv = enc.encode(&readings);
        assert_eq!(pv.vector.len(), 8);
    }

    #[test]
    fn test_encode_learned_projection() {
        let enc = PerceptionEncoder::new(3, 10, EncodingMethod::LearnedProjection);
        let readings = vec![SensorReading::new("temp", 5.0, "C")];
        let pv = enc.encode(&readings);
        assert_eq!(pv.vector.len(), 10);
    }

    #[test]
    fn test_batch_encoding() {
        let enc = PerceptionEncoder::new(3, 6, EncodingMethod::Raw);
        let batch_readings = vec![
            vec![SensorReading::new("temp", 10.0, "C")],
            vec![SensorReading::new("humidity", 50.0, "%")],
        ];
        let batch = enc.encode_batch(&batch_readings);
        assert_eq!(batch.vectors.len(), 2);
        assert_eq!(batch.encoding_method, EncodingMethod::Raw);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let v1 = make_pv(vec![1.0, 2.0, 3.0]);
        let v2 = make_pv(vec![1.0, 2.0, 3.0]);
        let sim = PerceptionVector::cosine_similarity(&v1, &v2);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let v1 = make_pv(vec![1.0, 0.0]);
        let v2 = make_pv(vec![0.0, 1.0]);
        let sim = PerceptionVector::cosine_similarity(&v1, &v2);
        assert!(sim.abs() < 1e-10);
    }

    #[test]
    fn test_euclidean_distance() {
        let v1 = make_pv(vec![0.0, 0.0]);
        let v2 = make_pv(vec![3.0, 4.0]);
        let dist = PerceptionVector::euclidean(&v1, &v2);
        assert!((dist - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_nearest_neighbor() {
        let query = make_pv(vec![1.0, 1.0]);
        let batch = PerceptionBatch {
            vectors: vec![
                make_pv(vec![10.0, 10.0]),
                make_pv(vec![1.1, 1.1]),
                make_pv(vec![5.0, 5.0]),
                make_pv(vec![1.0, 1.0]),
            ],
            encoding_method: EncodingMethod::Raw,
        };
        let nearest = batch.nearest(&query, 2);
        assert_eq!(nearest.len(), 2);
        assert!((nearest[0].vector[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalization() {
        let readings = vec![
            SensorReading::new("temp", 10.0, "C"),
            SensorReading::new("temp", 20.0, "C"),
            SensorReading::new("temp", 30.0, "C"),
        ];
        let normed = normalize_readings(&readings);
        assert_eq!(normed.len(), 3);
        let mean: f64 = normed.iter().map(|r| r.value).sum::<f64>() / 3.0;
        assert!(mean.abs() < 1e-10);
        let std: f64 = normed
            .iter()
            .map(|r| r.value.powi(2))
            .sum::<f64>()
            / 3.0;
        assert!((std - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_different_dimensions() {
        let enc = PerceptionEncoder::new(3, 100, EncodingMethod::Raw);
        let readings = vec![SensorReading::new("temp", 1.0, "C")];
        let pv = enc.encode(&readings);
        assert_eq!(pv.vector.len(), 100);
    }

    #[test]
    fn test_batch_stats() {
        let batch = PerceptionBatch {
            vectors: vec![make_pv(vec![3.0, 4.0]), make_pv(vec![6.0, 8.0])],
            encoding_method: EncodingMethod::Raw,
        };
        let stats = batch.stats();
        assert_eq!(stats.count, 2);
        assert_eq!(stats.dimension, 2);
        assert!((stats.avg_norm - 7.5).abs() < 1e-10); // (5+10)/2
    }

    #[test]
    fn test_empty_readings() {
        let enc = PerceptionEncoder::new(3, 6, EncodingMethod::Raw);
        let pv = enc.encode(&[]);
        assert_eq!(pv.vector.len(), 6);
        assert!(pv.vector.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_single_reading() {
        let enc = PerceptionEncoder::new(3, 9, EncodingMethod::Raw);
        let readings = vec![SensorReading::new("temp", 42.0, "C")];
        let pv = enc.encode(&readings);
        assert_eq!(pv.vector.len(), 9);
        assert_eq!(pv.vector[0], 42.0);
    }

    #[test]
    fn test_identical_readings() {
        let enc = PerceptionEncoder::new(3, 6, EncodingMethod::Normalized);
        let readings = vec![
            SensorReading::new("temp", 5.0, "C"),
            SensorReading::new("temp", 5.0, "C"),
        ];
        let pv = enc.encode(&readings);
        // With identical values, std_dev=0, values unchanged
        assert_eq!(pv.source_readings[0].value, 5.0);
    }

    #[test]
    fn test_very_large_values() {
        let enc = PerceptionEncoder::new(3, 3, EncodingMethod::Raw);
        let readings = vec![SensorReading::new("temp", 1e15, "C")];
        let pv = enc.encode(&readings);
        assert!((pv.vector[0] - 1e15).abs() < 1.0);
    }

    #[test]
    fn test_encoding_determinism() {
        let enc = PerceptionEncoder::new(3, 8, EncodingMethod::HashProjection);
        let readings = vec![
            SensorReading::new("temp", 25.0, "C"),
            SensorReading::new("humidity", 60.0, "%"),
        ];
        let v1 = enc.encode(&readings);
        let v2 = enc.encode(&readings);
        assert_eq!(v1.vector, v2.vector);
    }

    fn make_pv(v: Vec<f64>) -> PerceptionVector {
        PerceptionVector {
            id: Uuid::new_v4(),
            vector: v,
            source_readings: vec![],
            room_id: String::new(),
            timestamp: 0,
        }
    }
}
