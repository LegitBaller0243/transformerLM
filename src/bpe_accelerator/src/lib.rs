use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
mod bpe_accelerator {
    use pyo3::prelude::*;
    use std::collections::HashMap;
    use std::collections::HashSet;
    use pyo3::types::PyBytes;
    use fancy_regex::Regex;
    use std::cmp::Reverse;


    struct NodeTrain {
        token: u32, 
        prev: Option<usize>, 
        next: Option<usize>,
        merged: bool, 
        freq: usize,
    }

    struct NodeEncode {
        token: u32, 
        prev: Option<usize>, 
        next: Option<usize>,
        merged: bool, 
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
    fn train(py: Python<'_>, pre_tokens: HashMap<String, usize>, vocab_size: usize, special_tokens: Vec<String>) -> PyResult<(HashMap<u32, Py<PyBytes>>, Vec<(Py<PyBytes>, Py<PyBytes>)>)> {       
        let mut counts: HashMap<(u32, u32), usize> = HashMap::new();
        let mut merges = Vec::new(); 
        let mut vocab = HashMap::new(); //u32 -> Vec<u8>
        let mut curr_word = 256;


        let mut occurences: HashMap<(u32, u32), HashSet<usize>> = HashMap::new(); // A, B -> Occurences
        let mut backing: Vec<NodeTrain> = Vec::new();
        let mut heap_counts = std::collections::BinaryHeap::new();

        // so for every token in tokens, 
            // for every consecutive pair of bytes, update frequency in counts
        let mut index = 0;

        let mut rst_absorption: HashSet<usize> = HashSet::new();
        let mut changed_count_keys: HashSet<(u32, u32)> = HashSet::new();

        for i in 0..256 {
            vocab.insert(i, vec![i as u8]);
        }
        for special_token in special_tokens {
            vocab.insert(curr_word, special_token.as_bytes().to_vec());
            curr_word += 1;
        }

        for (token, freq) in &pre_tokens {
            let mut prv = None;
            for window in token.as_bytes().windows(2).map(|w| (w[0] as u32, w[1] as u32)) {
                let key = window.clone();
                *counts.entry(key).or_insert(0) += freq;
                //create a new Node for window; add to LinkedList and update occurences.

                
                let node = NodeTrain {
                    token: window.0, 
                    prev: prv, 
                    next: None, 
                    merged: false, 
                    freq: *freq,
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
            let node = NodeTrain {
                token: *token.as_bytes().last().unwrap_or(&0) as u32, 
                prev: prv, 
                next: None, 
                merged: false,
                freq: *freq,
            };
            if let Some(prev_index) = prv {
                backing[prev_index].next = Some(index);
            };
            index += 1;
            backing.push(node)
        }

        for (key, freq) in &counts {
            let (bytes1, bytes2) = (&vocab[&key.0], &vocab[&key.1]);
            heap_counts.push((*freq, (bytes1.clone(), bytes2.clone()), key.clone()));
        }

        let mut elapsed = std::time::Duration::new(0, 0);
        loop {
        // start loop
            if vocab.len() >= vocab_size {
                break;
            }

            let time_of_maximum_check = std::time::Instant::now();
            let mut my_vec = None;
            while heap_counts.len() > 0 {
                let (freq, _, key) = heap_counts.pop().unwrap();
                if counts.get(&key) == Some(&freq) {
                    my_vec = Some(key);
                    break;
                }
            }
            let my_vec = match my_vec {
                Some(v) => v,
                None => break,
            };

            elapsed += time_of_maximum_check.elapsed();


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
                *counts.entry(my_vec).or_insert(0) -= backing[occ].freq;

                            
                
                let new_curr_token = curr_word - 1; // index into vocab


                if let Some(prev_index) = node_prev {
                    let prev_token = backing[prev_index].token;
                    if let Some(occ_set) = occurences.get_mut(&(prev_token, curr_token)) {
                        occ_set.remove(&prev_index);
                    }
                    *counts.entry((prev_token, curr_token)).or_insert(0) -= backing[occ].freq;
                    changed_count_keys.insert((prev_token, curr_token));

                    backing[prev_index].next = Some(occ);

                    *counts.entry((prev_token, new_curr_token)).or_insert(0) += backing[occ].freq;
                    changed_count_keys.insert((prev_token, new_curr_token));
                    occurences.entry((prev_token, new_curr_token)).or_insert_with(HashSet::new).insert(prev_index);
                };
                    
                
                let next_token = backing[next_index].token;
                backing[next_index].merged = true;
                if let Some(nxt_nxt) = backing[next_index].next {
                    let nxt_nxt_token = backing[nxt_nxt].token;

                    if let Some(occ_set) = occurences.get_mut(&(next_token, nxt_nxt_token)) {
                        occ_set.remove(&next_index);
                    }
                    *counts.entry((next_token, nxt_nxt_token)).or_insert(0) -= backing[occ].freq;
                    changed_count_keys.insert((next_token, nxt_nxt_token));

                    backing[nxt_nxt].prev = Some(occ);
                    backing[occ].next = Some(nxt_nxt);


                    *counts.entry((new_curr_token, nxt_nxt_token)).or_insert(0) += backing[occ].freq;
                    changed_count_keys.insert((new_curr_token, nxt_nxt_token));
                    occurences.entry((new_curr_token, nxt_nxt_token)).or_insert_with(HashSet::new).insert(occ);                        
                } else {
                    backing[occ].next = None;
                }
                backing[occ].token = new_curr_token;
                backing[occ].merged = true;
                rst_absorption.insert(occ);
            }
            // iterate through changed_count_keys and update heap_counts with the byte tuple and freq
            for key in &changed_count_keys {
                let freq = counts.get(key).unwrap_or(&0);
                if freq == &0 {
                    continue;
                }
                let (bytes1, bytes2) = (&vocab[&key.0], &vocab[&key.1]);
                heap_counts.push((*freq, (bytes1.clone(), bytes2.clone()), key.clone()));
            }
            changed_count_keys.clear();
            
            // delete the myvec entry from counts
            counts.remove(&my_vec);

        }
        println!("Elapsed time for maximum key check: {:?}", elapsed); // put exclamation point to print to stdout
        convert_to_py_types(py, vocab, merges)
    }
    fn merge_fast(
        token_bytes: &[u8],
        encode_vocab: &HashMap<Vec<u8>, u32>,
        merges: &Vec<(Vec<u8>, Vec<u8>)>, 
        merge_dict: &HashMap<(u32, u32), usize>
    ) 
    -> Vec<u32> {
        let mut parts: Vec<NodeEncode> = Vec::new();
        let mut merge_candidates = std::collections::BinaryHeap::new();
        let mut final_ids: Vec<u32> = Vec::new();
        // parts needs to have prev and. next pointers and a merged flag and freq count
        // look at token btes in windows - do the same thing 
        let mut prv = None;
        let mut index = 0;
        for window in token_bytes.windows(2).map(|w| (encode_vocab.get(&vec![w[0]]).cloned(), encode_vocab.get(&vec![w[1]]).cloned())) {
            let tok1_id = window.0.unwrap();
            let tok2_id = window.1.unwrap();

            let node = NodeEncode {
                token: tok1_id,
                prev: prv, 
                next: None, 
                merged: false, 
            };
            if let Some(prev_index) = prv {
                parts[prev_index].next = Some(index);
            };
            parts.push(node);
            prv = Some(index);
            index += 1;
            // each byte-pair if possible merge needs to be added to merge min heap
            let merge_opt = merge_dict.get(&(tok1_id, tok2_id)).cloned();
            if let Some(merge_index) = merge_opt {
                merge_candidates.push(Reverse((merge_index, (index - 1, index), (tok1_id, tok2_id))));
            }
        }
        let last_byte_token = encode_vocab.get(&vec![*token_bytes.last().unwrap()]).cloned();
        let node = NodeEncode {
            token: last_byte_token.unwrap(),
            prev: prv,
            next: None,
            merged: false,
        };
        if let Some(prev_index) = prv {
            parts[prev_index].next = Some(index);
        };
        parts.push(node); 

        while let Some(Reverse((merge_index, (i, j), (tok1_id, tok2_id)))) = merge_candidates.pop() {
            if parts[i].next != Some(j) {
                continue;
            }
            //check if lens are same 
            if parts[i].token != tok1_id || parts[j].token != tok2_id {
                continue;
            }
            if parts[i].merged || parts[j].merged {
                continue;
            }
            // all g to merged
            let mut new_token = merges[merge_index].0.clone();
            new_token.extend(&merges[merge_index].1);
            let new_token_id = *encode_vocab.get(&new_token).unwrap();
            //new token combine , update prev next etc. 
            // change parts[i].token, .next
            parts[i].token = new_token_id;
            parts[i].next = parts[j].next;
            parts[j].merged = true;
            if let Some(next_index) = parts[j].next {
                parts[next_index].prev = Some(i);
                
                if let Some(&priority) = merge_dict.get(&(new_token_id, parts[next_index].token)) {
                    merge_candidates.push(Reverse((priority, (i, next_index), (new_token_id, parts[next_index].token))));
                }
            }
            if parts[i].prev.is_some() {
                let prev_index = parts[i].prev.unwrap();
                parts[prev_index].next = Some(i);
                if let Some(&priority) = merge_dict.get(&(parts[prev_index].token, new_token_id)) {
                    merge_candidates.push(Reverse((priority, (prev_index, i), (parts[prev_index].token, new_token_id))));
                }
            }
            //add new combos so check prev + new token and new token + next, add to heap.    
        }
        for part in &parts {
            if !part.merged {
                final_ids.push(part.token);
            }
        }
        return final_ids;
    }

    #[pyfunction]
    // takes in a list str, merges dict, vocab dict
    fn encode_(
        pre_tokens: Vec<String>,
        encode_vocab: HashMap<Vec<u8>, u32>,
        merges: Vec<(Vec<u8>, Vec<u8>)>, 
        special_tokens: HashSet<String>
    ) 
    -> Vec<u32> {
        // create list of final_ids
        let mut final_ids: Vec<u32> = Vec::new();
        let gpt_split_pattern = Regex::new(
            r#"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#
        ).unwrap(); 
    
        let mut merge_dict: HashMap<(u32, u32), usize> = HashMap::new();
        //
        for (i, (b1, b2)) in merges.iter().enumerate() {
            let token1 = encode_vocab[b1];
            let token2 = encode_vocab[b2];
            merge_dict.insert((token1, token2), i);
        }
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
                    let curr_ids = merge_fast(chunk.as_bytes(), &encode_vocab, &merges, &merge_dict); 
                    final_ids.extend(&curr_ids);
                }
            }
        }
        //return list of token_ids. 
        return final_ids
    }
}


