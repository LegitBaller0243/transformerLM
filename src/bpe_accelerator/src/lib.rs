use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
mod bpe_accelerator {
    use pyo3::prelude::*;
    use std::collections::HashMap;
    use std::collections::HashSet;
    use std::cmp::Ordering;
    use pyo3::types::PyBytes;
    use fancy_regex::Regex;


    struct Node {
        token: u32, 
        prev: Option<usize>, 
        next: Option<usize>,
        merged: bool
    }

    fn convert_to_py_types(py: Python<'_>, vocab: HashMap<u32, Vec<u8>>, merges: Vec<(Vec<u8>, Vec<u8>)>) -> PyResult<(HashMap<u32, Py<PyBytes>>, Vec<(Py<PyBytes>, Py<PyBytes>)>)> {
        let mut py_vocab = HashMap::with_capacity(vocab.len());
        for (id, byte_vec) in vocab {
            let py_bytes: Py<PyBytes> = PyBytes::new(py, &byte_vec).into();
            py_vocab.insert(id, py_bytes);
        }

        let mut py_merges = Vec::with_capacity(merges.len());
        for (b1, b2) in merges {
            let py_b1: Py<PyBytes> = PyBytes::new(py, &b1).into();
            let py_b2: Py<PyBytes> = PyBytes::new(py, &b2).into();
            py_merges.push((py_b1, py_b2));
        }
        Ok((py_vocab, py_merges))
    }



    /// return vocab: dict[int, bytes] and merges: list[tuple[bytes, bytes]]
    #[pyfunction]
    fn train(py: Python<'_>, pre_tokens: Vec<String>, vocab_size: usize,) -> PyResult<(HashMap<u32, Py<PyBytes>>, Vec<(Py<PyBytes>, Py<PyBytes>)>)> {       
        let mut counts: HashMap<(u32, u32), usize> = HashMap::new();
        let mut merges = Vec::new(); 
        let mut vocab = HashMap::new(); //u32 -> Vec<u8>
        let mut curr_word = 256;

        let mut occurences: HashMap<(u32, u32), HashSet<usize>> = HashMap::new(); // A, B -> Occurences
        let mut backing: Vec<Node> = Vec::new();

        // so for every token in tokens, 
            // for every consecutive pair of bytes, update frequency in counts
        let mut index = 0;

        let mut rst_absorption: HashSet<usize> = HashSet::new();

        for i in 0..256 {
            vocab.insert(i, vec![i as u8]);
        }

        for token in pre_tokens {
            let mut prv = None;
            for window in token.as_bytes().windows(2).map(|w| (w[0] as u32, w[1] as u32)) {
                let key = window.clone();
                *counts.entry(key).or_insert(0) += 1;
                //create a new Node for window; add to LinkedList and update occurences.

                
                let node = Node {
                    token: window.0, 
                    prev: prv, 
                    next: None, 
                    merged: false
                };
                if let Some(prev_index) = prv {
                    backing[prev_index].next = Some(index);
                };
                backing.push(node);

                let key = window.clone();
                occurences.entry(key).or_insert_with(HashSet::new).insert(index);
                prv = Some(index);
                index += 1
                
                // each byte-pair needs to add this occurence to the occurences table
            }
            let node = Node {
                token: *token.as_bytes().last().unwrap_or(&0) as u32, 
                prev: prv, 
                next: None, 
                merged: false
            };
            if let Some(prev_index) = prv {
                backing[prev_index].next = Some(index);
            };
            index += 1;
            backing.push(node)
        }
        loop {
        // start loop
            if vocab.len() >= vocab_size {
                break;
            }


            let maximum = counts.iter().max_by(|(pair_a, freq_a), (pair_b, freq_b)| {
                match freq_a.cmp(freq_b) {
                    Ordering::Equal => {
                        let bytes_a = (&vocab[&pair_a.0], &vocab[&pair_a.1]);
                        let bytes_b = (&vocab[&pair_b.0], &vocab[&pair_b.1]);
                        bytes_a.cmp(&bytes_b)
                    }
                    other => other,
                }
            }); 

            // at best its a tuple of a byte-pair
            let my_vec;
            if let Some((pair, _)) = maximum {
                my_vec = *pair;
            } else {
                break;
            }

            if counts[&my_vec] == 0 {
                break;
            }
            for &rst in &rst_absorption {
                backing[rst].merged = false;
            }
            rst_absorption.clear();

            // merge should hold Vec<8>'s
            let bytes_a = &vocab[&my_vec.0];
            let bytes_b = &vocab[&my_vec.1];
            merges.push((bytes_a.clone(), bytes_b.clone()));

            let left_bytes = &vocab[&my_vec.0];
            let right_bytes = &vocab[&my_vec.1];

            let mut new_token: Vec<u8> = Vec::with_capacity(left_bytes.len() + right_bytes.len());
            new_token.extend_from_slice(left_bytes);
            new_token.extend_from_slice(right_bytes);

            vocab.insert(curr_word, new_token);
            curr_word += 1;
            // add to merges + vocab
            
            let occs_set = occurences.remove(&my_vec).expect("We use try to merge all of the occurences");
            let mut occs_vec: Vec<usize> = occs_set.into_iter().collect();
            occs_vec.sort();

            for &occ in &occs_vec {
                let node_merged = backing[occ].merged;
                if node_merged {
                    continue;
                }
                let curr_token = backing[occ].token; // index into vocab
                let node_prev = backing[occ].prev; //indices into backing
                let node_next = backing[occ].next;

                let next_index = node_next.expect("Expected next index because of occurences (l, r) = l invariant");
                
                if backing[next_index].merged {
                    continue;
                }
                *counts.entry(my_vec).or_insert(0) -= 1;

                            
                
                let new_curr_token = curr_word - 1; // index into vocab


                if let Some(prev_index) = node_prev {
                    let prev_token = backing[prev_index].token;
                    if let Some(occ_set) = occurences.get_mut(&(prev_token, curr_token)) {
                        occ_set.remove(&prev_index);
                    }
                    *counts.entry((prev_token, curr_token)).or_insert(0) -= 1;

                    backing[prev_index].next = Some(occ);

                    *counts.entry((prev_token, new_curr_token)).or_insert(0) += 1;
                    occurences.entry((prev_token, new_curr_token)).or_insert_with(HashSet::new).insert(prev_index);
                };
                    
                
                let next_token = backing[next_index].token;
                backing[next_index].merged = true;
                if let Some(nxt_nxt) = backing[next_index].next {
                    let nxt_nxt_token = backing[nxt_nxt].token;

                    if let Some(occ_set) = occurences.get_mut(&(next_token, nxt_nxt_token)) {
                        occ_set.remove(&next_index);
                    }
                    *counts.entry((next_token, nxt_nxt_token)).or_insert(0) -= 1;

                    backing[nxt_nxt].prev = Some(occ);
                    backing[occ].next = Some(nxt_nxt);


                    *counts.entry((new_curr_token, nxt_nxt_token)).or_insert(0) += 1;
                    occurences.entry((new_curr_token, nxt_nxt_token)).or_insert_with(HashSet::new).insert(occ);                        
                } else {
                    backing[occ].next = None;
                }
                backing[occ].token = new_curr_token;
                backing[occ].merged = true;
                rst_absorption.insert(occ);
            }
            // delete the myvec entry from counts
            counts.remove(&my_vec);
        }
        convert_to_py_types(py, vocab, merges)
    }
    fn merge_(
        token_bytes: &[u8],
        encode_vocab: &HashMap<Vec<u8>, u32>,
        merges: &Vec<(Vec<u8>, Vec<u8>)>) 
    -> Vec<u32> {
        let mut token_ids: Vec<u32> = Vec::new();
        // i need to turn the token_bytes ref into a list so i can combine elements quickly
        let mut parts: Vec<Vec<u8>> = token_bytes.iter().map(|&b| vec![b]).collect();
        // for each token bytes - Vec<u8>

        for j in 0..merges.len() {
            let mut i = 0;
            let first = &merges[j].0;
            let second = &merges[j].1;

            let mut new_parts = Vec::new();
            while i < parts.len() {
             // check if the merge is applicable to the current pair of token_bytes
             // combine then keep going
                if i < parts.len() - 1  && &parts[i] == first && &parts[i + 1] == second {
                    let mut combined = parts[i].clone();
                    combined.extend(&parts[i + 1]);
                    new_parts.push(combined);
                    i += 2;
                } else {
                    new_parts.push(parts[i].clone());
                    i += 1;
                }
            }
            parts = new_parts;
        }
        for j in 0..parts.len() {
            let token_chunk: Vec<u8> = parts[j].clone();
            token_ids.push(encode_vocab[&token_chunk]);
        }
        return token_ids
    }

    #[pyfunction]
    // takes in a list str, merges dict, vocab dict
    fn encode(
        pre_tokens: Vec<String>,
        encode_vocab: HashMap<Vec<u8>, u32>,
        merges: Vec<(Vec<u8>, Vec<u8>)>, 
        special_tokens: HashSet<String>
    ) 
    -> Vec<u32> {
        // create list of final_ids
        let mut final_ids: Vec<u32> = Vec::new();
        let gpt_split_pattern = Regex::new(
            r#"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"#
        ).unwrap(); 
        // for segment in post_regex_split
        for token in pre_tokens {
            if special_tokens.contains(&token) {
                //encode special tokens
                final_ids.extend([encode_vocab[&token.as_bytes().to_vec()]]);
            } else {
                //pretokenize the segment
                let matches: Vec<String> = gpt_split_pattern.find_iter(&token).filter_map(|m| m.ok()).map(|m| m.as_str().to_string()).collect();
                //for each chunk in segment
                    //merge_(bytes) -> list of curr_token_ids
                    // final_ids.extend(curr_token_ids)
                for chunk in matches {
                    let curr_ids = merge_(chunk.as_bytes(), &encode_vocab, &merges);
                    final_ids.extend(&curr_ids)
                }
            }
        }
            //skip null segments IN PYTHON
            //python also needs to split special tokens
        return final_ids
        //return list of curr_token_ids. 
    }

}


