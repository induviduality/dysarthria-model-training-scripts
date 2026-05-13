import json
import re
from collections import defaultdict

# Soundex implementation for phonetic matching
def soundex(name):
    """Generate Soundex code for phonetic matching"""
    if not name:
        return ''
    
    name = name.upper()
    
    codes = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }
    
    soundex_code = name[0]
    
    for char in name[1:]:
        code = codes.get(char, '0')
        if code != '0' and code != soundex_code[-1]:
            soundex_code += code
    
    return (soundex_code + '000')[:4]

# Common contractions
CONTRACTIONS = {
    'dont': "don't",
    'cant': "can't",
    'wont': "won't",
    'ive': "i've",
    'youve': "you've",
    'theyve': "they've",
    'isnt': "isn't",
    'arent': "aren't",
    'wasnt': "wasn't",
    'werent': "weren't",
    'hasnt': "hasn't",
    'havent': "haven't",
    'shouldnt': "shouldn't",
    'wouldnt': "wouldn't",
    'couldnt': "couldn't",
    'hes': "he's",
    'shes': "she's",
    'its': "it's",
    'hed': "he'd",
    'shed': "she'd",
    'wed': "we'd",
    'theyd': "they'd",
    'ill': "i'll",
    'youll': "you'll",
    'hell': "he'll",
    'shell': "she'll",
    'well': "we'll",
    'theyll': "they'll",
}

# Number word to numeral mapping
NUMBERS = {
    'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
    'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
    'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
    'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
    'eighteen': '18', 'nineteen': '19', 'twenty': '20', 'thirty': '30',
    'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70',
    'eighty': '80', 'ninety': '90', 'hundred': '100', 'thousand': '1000',
    'ninetythree': '93',
}

# Mapping of speakers to their datasets
SPEAKER_TO_DATASET = {
    'F03': 'TORGO', 'F04': 'TORGO', 'M02': 'TORGO', 'M03': 'TORGO', 'M04': 'TORGO', 'M05': 'TORGO',
    'F02': 'UASpeech', 'F05': 'UASpeech', 'M01': 'UASpeech', 'M08': 'UASpeech', 'M09': 'UASpeech',
    'M10': 'UASpeech', 'M11': 'UASpeech', 'M12': 'UASpeech', 'M14': 'UASpeech', 'M16': 'UASpeech',
}

def normalize_text(text):
    """Normalize text: expand contractions, handle hyphenation"""
    text = text.lower()
    
    for contraction, expanded in CONTRACTIONS.items():
        text = re.sub(r'\b' + contraction + r'\b', expanded.replace("'", ""), text)
    
    text = text.replace('-', ' ')
    return text

def normalize_word(word):
    """Remove punctuation and lowercase"""
    word = re.sub(r'[^\w]', '', word).lower()
    return word

def words_match(ref_word, hyp_word):
    """Check if two words match"""
    ref_normalized = normalize_word(ref_word)
    hyp_normalized = normalize_word(hyp_word)
    
    if not ref_normalized or not hyp_normalized:
        return ref_normalized == hyp_normalized
    
    if ref_normalized == hyp_normalized:
        return True
    
    if soundex(ref_normalized) == soundex(hyp_normalized):
        return True
    
    if ref_normalized in NUMBERS:
        ref_as_num = NUMBERS[ref_normalized]
        if ref_as_num == hyp_normalized:
            return True
    
    if hyp_normalized in NUMBERS:
        hyp_as_num = NUMBERS[hyp_normalized]
        if ref_normalized == hyp_as_num:
            return True
    
    return False

def calculate_correct_words(reference, hypothesis):
    """Calculate correct words using our matching rules"""
    ref_normalized = normalize_text(reference)
    hyp_normalized = normalize_text(hypothesis)
    
    ref_words = ref_normalized.split()
    hyp_words = hyp_normalized.split()
    
    correct_count = 0
    hyp_idx = 0
    
    for ref_word in ref_words:
        while hyp_idx < len(hyp_words):
            if words_match(ref_word, hyp_words[hyp_idx]):
                correct_count += 1
                hyp_idx += 1
                break
            hyp_idx += 1
    
    return correct_count, len(ref_words)

def analyze_model(filepath, model_name):
    """Analyze model using our word-matching rules"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    per_sample = data.get('per_sample', [])
    
    # Group results by speaker and dataset
    dataset_stats = {}
    speaker_stats = {}
    
    for sample in per_sample:
        reference = sample.get('reference', '')
        hypothesis = sample.get('hypothesis', '')
        
        # Try to find speaker from the data (if available)
        # For validation set, we'll group by speaker later through metadata
        correct, total = calculate_correct_words(reference, hypothesis)
        
        # Store per-sample result for later analysis
        if 'speaker_id' not in sample:
            # We need to infer speaker from per_speaker_wer
            sample['correct'] = correct
            sample['total'] = total
    
    # Now group by speaker using per_speaker_wer
    per_speaker_wer = data.get('per_speaker_wer', {})
    
    for speaker_id in per_speaker_wer.keys():
        dataset = SPEAKER_TO_DATASET.get(speaker_id, 'Unknown')
        
        if dataset not in dataset_stats:
            dataset_stats[dataset] = {'correct': 0, 'total': 0, 'samples': 0, 'speakers': []}
        
        speaker_stats[speaker_id] = {'correct': 0, 'total': 0, 'samples': 0}
    
    # Iterate through per_sample and match to speakers based on count
    # We need to split samples by speaker - let's use n_samples from per_speaker_wer
    sample_idx = 0
    
    for speaker_id, speaker_info in sorted(per_speaker_wer.items()):
        n_samples = speaker_info.get('n_samples', 0)
        dataset = SPEAKER_TO_DATASET.get(speaker_id, 'Unknown')
        
        speaker_correct = 0
        speaker_total = 0
        
        # Process samples for this speaker
        for i in range(n_samples):
            if sample_idx < len(per_sample):
                sample = per_sample[sample_idx]
                reference = sample.get('reference', '')
                hypothesis = sample.get('hypothesis', '')
                
                correct, total = calculate_correct_words(reference, hypothesis)
                speaker_correct += correct
                speaker_total += total
                
                sample_idx += 1
        
        # Store speaker stats
        speaker_stats[speaker_id] = {
            'correct': speaker_correct,
            'total': speaker_total,
            'samples': n_samples,
            'accuracy': (speaker_correct / speaker_total * 100) if speaker_total > 0 else 0
        }
        
        # Aggregate to dataset
        dataset_stats[dataset]['correct'] += speaker_correct
        dataset_stats[dataset]['total'] += speaker_total
        dataset_stats[dataset]['samples'] += n_samples
        if speaker_id not in dataset_stats[dataset]['speakers']:
            dataset_stats[dataset]['speakers'].append(speaker_id)
    
    # Print results
    print(f"\n{'='*110}")
    print(f"EVALUATION RESULTS BY DATASET SOURCE - {model_name}")
    print(f"(Using word-matching rules: phonetics, punctuation, hyphens, numbers)")
    print(f"{'='*110}\n")
    
    overall_correct = sum(d['correct'] for d in dataset_stats.values())
    overall_total = sum(d['total'] for d in dataset_stats.values())
    overall_acc = (overall_correct / overall_total * 100) if overall_total > 0 else 0
    
    print(f"Overall Performance:")
    print(f"  Accuracy: {overall_acc:.2f}%")
    print(f"  Correct words: {overall_correct}/{overall_total}")
    print(f"  Total samples: {data.get('n_samples', 'N/A')}")
    print(f"  Model: {data.get('model_path', 'N/A')}")
    
    print(f"\n{'='*110}\n")
    print(f"{'Dataset':<15} {'Accuracy':<12} {'Correct Words':<20} {'Total Samples':<15}")
    print(f"{'-'*110}")
    
    dataset_acc_map = {}
    
    for dataset in sorted(dataset_stats.keys()):
        stats = dataset_stats[dataset]
        acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        dataset_acc_map[dataset] = acc
        
        print(f"{dataset:<15} {acc:>10.2f}% {stats['correct']:>8}/{stats['total']:<8} {stats['samples']:>10} samples")
    
    print(f"\n{'='*110}\n")
    print("PER-SPEAKER ACCURACY BREAKDOWN:")
    print(f"{'='*110}\n")
    
    print(f"{'Speaker':<10} {'Dataset':<15} {'Accuracy':<12} {'Correct Words':<20} {'Samples':<10}")
    print(f"{'-'*110}")
    
    for speaker_id in sorted(speaker_stats.keys()):
        stats = speaker_stats[speaker_id]
        dataset = SPEAKER_TO_DATASET.get(speaker_id, 'Unknown')
        
        print(f"{speaker_id:<10} {dataset:<15} {stats['accuracy']:>10.2f}% {stats['correct']:>8}/{stats['total']:<8} {stats['samples']:>6}")
    
    return dataset_stats, speaker_stats, dataset_acc_map

def main():
    print("\n" + "="*110)
    print("VALIDATION SET ANALYSIS - BY DATASET SOURCE")
    print("(Using phonetic, punctuation, hyphenation, and number normalization)")
    print("="*110)
    
    # Analyze base model
    base_dataset_stats, base_speaker_stats, base_dataset_acc = analyze_model(
        'inference-outputs/evaluation_results_base.json', 'BASE MODEL'
    )

    # Analyze finetuned model
    tuned_dataset_stats, tuned_speaker_stats, tuned_dataset_acc = analyze_model(
        'inference-outputs/evaluation_results.json', 'FINETUNED MODEL'
    )
    
    # Comparison
    print(f"\n{'='*110}")
    print("IMPROVEMENT COMPARISON (Finetuned vs Base)")
    print(f"{'='*110}\n")
    
    print(f"{'Dataset':<15} {'Base Acc':<12} {'Tuned Acc':<12} {'Improvement':<15} {'Direction':<15}")
    print(f"{'-'*110}")
    
    for dataset in sorted(set(base_dataset_acc.keys()) | set(tuned_dataset_acc.keys())):
        if dataset in base_dataset_acc and dataset in tuned_dataset_acc:
            base_acc = base_dataset_acc[dataset]
            tuned_acc = tuned_dataset_acc[dataset]
            improvement = tuned_acc - base_acc
            direction = "✓ Better" if improvement > 0 else "✗ Worse" if improvement < 0 else "="
            
            print(f"{dataset:<15} {base_acc:>10.2f}% {tuned_acc:>10.2f}% {improvement:>+8.2f}pp    {direction:<15}")

if __name__ == '__main__':
    main()
