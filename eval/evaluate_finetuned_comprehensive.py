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
    'dont': "don't", 'cant': "can't", 'wont': "won't", 'ive': "i've",
    'youve': "you've", 'theyve': "they've", 'isnt': "isn't", 'arent': "aren't",
    'wasnt': "wasn't", 'werent': "weren't", 'hasnt': "hasn't", 'havent': "haven't",
    'shouldnt': "shouldn't", 'wouldnt': "wouldn't", 'couldnt': "couldn't",
    'hes': "he's", 'shes': "she's", 'its': "it's", 'hed': "he'd",
    'shed': "she'd", 'wed': "we'd", 'theyd': "they'd", 'ill': "i'll",
    'youll': "you'll", 'hell': "he'll", 'shell': "she'll", 'well': "we'll", 'theyll': "they'll",
}

# Number word to numeral mapping
NUMBERS = {
    'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
    'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
    'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14', 'fifteen': '15',
    'sixteen': '16', 'seventeen': '17', 'eighteen': '18', 'nineteen': '19', 'twenty': '20',
    'thirty': '30', 'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70',
    'eighty': '80', 'ninety': '90', 'hundred': '100', 'thousand': '1000', 'ninetythree': '93',
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
        if NUMBERS[ref_normalized] == hyp_normalized:
            return True
    
    if hyp_normalized in NUMBERS:
        if ref_normalized == NUMBERS[hyp_normalized]:
            return True
    
    return False

def calculate_correct_words(reference, hypothesis):
    """Calculate correct words using matching rules"""
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

def main():
    # Load evaluation results
    with open('inference-outputs/evaluation_results.json', 'r') as f:
        data = json.load(f)
    
    per_sample = data.get('per_sample', [])
    per_speaker_wer = data.get('per_speaker_wer', {})
    
    # Initialize statistics
    holistic_correct = 0
    holistic_total = 0
    
    dataset_stats = {}
    speaker_stats = {}
    
    # Initialize dataset stats
    for speaker_id in per_speaker_wer.keys():
        dataset = SPEAKER_TO_DATASET.get(speaker_id, 'Unknown')
        if dataset not in dataset_stats:
            dataset_stats[dataset] = {'correct': 0, 'total': 0, 'samples': 0, 'speakers': []}
        if speaker_id not in dataset_stats[dataset]['speakers']:
            dataset_stats[dataset]['speakers'].append(speaker_id)
    
    # Process samples by speaker
    sample_idx = 0
    for speaker_id in sorted(per_speaker_wer.keys()):
        n_samples = per_speaker_wer[speaker_id].get('n_samples', 0)
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
                holistic_correct += correct
                holistic_total += total
                
                sample_idx += 1
        
        # Store speaker stats
        speaker_stats[speaker_id] = {
            'correct': speaker_correct,
            'total': speaker_total,
            'samples': n_samples,
            'accuracy': (speaker_correct / speaker_total * 100) if speaker_total > 0 else 0,
            'dataset': dataset
        }
        
        # Aggregate to dataset
        dataset_stats[dataset]['correct'] += speaker_correct
        dataset_stats[dataset]['total'] += speaker_total
        dataset_stats[dataset]['samples'] += n_samples
    
    # Calculate accuracies
    holistic_accuracy = (holistic_correct / holistic_total * 100) if holistic_total > 0 else 0
    
    # Print results
    print(f"\n{'='*110}")
    print(f"FINETUNED MODEL EVALUATION - COMPREHENSIVE ANALYSIS")
    print(f"(Using word-matching rules: phonetics, punctuation, hyphens, numbers)")
    print(f"{'='*110}\n")
    
    # HOLISTIC RESULTS
    print(f"{'HOLISTIC RESULTS (Overall):':^110}")
    print(f"{'-'*110}")
    print(f"Total Accuracy:           {holistic_accuracy:.2f}%")
    print(f"Correct Words:            {holistic_correct}/{holistic_total}")
    print(f"Total Samples:            {data.get('n_samples', 'N/A')}")
    print(f"Model:                    {data.get('model_path', 'N/A')}")
    print(f"Split:                    {data.get('split', 'N/A')}")
    
    # DATASET-WISE RESULTS
    print(f"\n{'='*110}")
    print(f"{'DATASET-WISE RESULTS':^110}")
    print(f"{'='*110}\n")
    
    print(f"{'Dataset':<15} {'Accuracy':<12} {'Correct Words':<20} {'Total Samples':<20}")
    print(f"{'-'*110}")
    
    for dataset in sorted(dataset_stats.keys()):
        stats = dataset_stats[dataset]
        acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        
        print(f"{dataset:<15} {acc:>10.2f}% {stats['correct']:>8}/{stats['total']:<8} {stats['samples']:>10} samples")
    
    # PER-SPEAKER RESULTS
    print(f"\n{'='*110}")
    print(f"{'PER-SPEAKER RESULTS':^110}")
    print(f"{'='*110}\n")
    
    print(f"{'Speaker':<10} {'Dataset':<15} {'Accuracy':<12} {'Correct Words':<20} {'Samples':<10}")
    print(f"{'-'*110}")
    
    for speaker_id in sorted(speaker_stats.keys()):
        stats = speaker_stats[speaker_id]
        print(f"{speaker_id:<10} {stats['dataset']:<15} {stats['accuracy']:>10.2f}% {stats['correct']:>8}/{stats['total']:<8} {stats['samples']:>6}")
    
    # DATASET SUMMARY TABLE
    print(f"\n{'='*110}")
    print(f"{'DATASET SUMMARY':^110}")
    print(f"{'='*110}\n")
    
    print(f"{'Dataset':<15} {'# Speakers':<15} {'# Samples':<15} {'Avg Accuracy':<15} {'Total Words':<15}")
    print(f"{'-'*110}")
    
    for dataset in sorted(dataset_stats.keys()):
        stats = dataset_stats[dataset]
        acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        
        print(f"{dataset:<15} {len(stats['speakers']):<15} {stats['samples']:<15} {acc:>10.2f}% {stats['total']:<15}")
    
    # SPEAKER RANKING BY ACCURACY
    print(f"\n{'='*110}")
    print(f"{'SPEAKER RANKING (by Accuracy)':^110}")
    print(f"{'='*110}\n")
    
    sorted_speakers = sorted(speaker_stats.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    print(f"{'Rank':<6} {'Speaker':<10} {'Dataset':<15} {'Accuracy':<12} {'Correct Words':<20}")
    print(f"{'-'*110}")
    
    for rank, (speaker_id, stats) in enumerate(sorted_speakers, 1):
        print(f"{rank:<6} {speaker_id:<10} {stats['dataset']:<15} {stats['accuracy']:>10.2f}% {stats['correct']:>8}/{stats['total']:<8}")
    
    # DETAILED STATISTICS
    print(f"\n{'='*110}")
    print(f"{'DETAILED STATISTICS':^110}")
    print(f"{'='*110}\n")
    
    accuracies = [s['accuracy'] for s in speaker_stats.values()]
    
    print(f"Speaker Accuracy Statistics:")
    print(f"  Highest:  {max(accuracies):.2f}%")
    print(f"  Lowest:   {min(accuracies):.2f}%")
    print(f"  Average:  {sum(accuracies)/len(accuracies):.2f}%")
    print(f"  Median:   {sorted(accuracies)[len(accuracies)//2]:.2f}%")
    
    # Performance by dataset
    for dataset in sorted(dataset_stats.keys()):
        stats = dataset_stats[dataset]
        acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        speakers_in_dataset = [s for s, info in speaker_stats.items() if info['dataset'] == dataset]
        accuracies_in_dataset = [speaker_stats[s]['accuracy'] for s in speakers_in_dataset]
        
        print(f"\n{dataset}:")
        print(f"  Accuracy:     {acc:.2f}%")
        print(f"  Speakers:     {len(speakers_in_dataset)}")
        print(f"  Avg Speaker Accuracy: {sum(accuracies_in_dataset)/len(accuracies_in_dataset):.2f}%")
        print(f"  Best Speaker: {max(speakers_in_dataset, key=lambda s: speaker_stats[s]['accuracy'])} ({max(accuracies_in_dataset):.2f}%)")
        print(f"  Worst Speaker: {min(speakers_in_dataset, key=lambda s: speaker_stats[s]['accuracy'])} ({min(accuracies_in_dataset):.2f}%)")

if __name__ == '__main__':
    main()
