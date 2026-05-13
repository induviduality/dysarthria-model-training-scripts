import json
import re
from difflib import SequenceMatcher

# Soundex implementation for phonetic matching
def soundex(name):
    """Generate Soundex code for phonetic matching (handles homophones like new/knew)"""
    name = name.upper()
    
    # Map letters to codes
    codes = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }
    
    soundex_code = name[0]  # Keep first letter
    
    for char in name[1:]:
        code = codes.get(char, '0')
        if code != '0' and code != soundex_code[-1]:  # Avoid duplicates
            soundex_code += code
    
    # Pad or truncate to 4 characters
    return (soundex_code + '000')[:4]

# Number word to numeral mapping
NUMBERS = {
    'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
    'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
    'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
    'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
    'eighteen': '18', 'nineteen': '19', 'twenty': '20', 'thirty': '30',
    'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70',
    'eighty': '80', 'ninety': '90', 'hundred': '100', 'thousand': '1000'
}

def normalize_word(word):
    """Remove punctuation and lowercase"""
    word = re.sub(r'[^\w]', '', word).lower()
    return word

def words_match(ref_word, hyp_word):
    """Check if two words match, considering phonetics and number conversions"""
    ref_normalized = normalize_word(ref_word)
    hyp_normalized = normalize_word(hyp_word)
    
    # Exact match
    if ref_normalized == hyp_normalized:
        return True
    
    # Check phonetic match (catches homophones like new/knew, to/too/two, etc.)
    if soundex(ref_normalized) == soundex(hyp_normalized):
        return True
    
    # Check number conversions (35 vs thirty-five)
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
    """Calculate how many words in reference are correctly transcribed in hypothesis"""
    # Split into words
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    correct_count = 0
    
    # For each reference word, try to find a match in hypothesis
    hyp_idx = 0
    for ref_word in ref_words:
        # Find the next matching word in hypothesis
        found = False
        while hyp_idx < len(hyp_words):
            if words_match(ref_word, hyp_words[hyp_idx]):
                correct_count += 1
                hyp_idx += 1
                found = True
                break
            hyp_idx += 1
        
        # If no match found, word is incorrect (don't increment correct_count)
    
    return correct_count, len(ref_words)

def main():
    # Load JSON
    with open('inference-outputs/evaluation_results.json', 'r') as f:
        data = json.load(f)
    
    per_sample = data.get('per_sample', [])
    
    total_correct = 0
    total_words = 0
    sample_results = []
    
    for idx, sample in enumerate(per_sample):
        reference = sample.get('reference', '')
        hypothesis = sample.get('hypothesis', '')
        
        correct, total = calculate_correct_words(reference, hypothesis)
        total_correct += correct
        total_words += total
        
        sample_results.append({
            'index': idx,
            'reference': reference,
            'hypothesis': hypothesis,
            'correct': correct,
            'total': total,
            'accuracy': f"{(correct/total*100):.1f}%" if total > 0 else "N/A"
        })
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"TRANSCRIPTION ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"Total samples: {len(per_sample)}")
    print(f"Total words: {total_words}")
    print(f"Correct words: {total_correct}")
    print(f"Overall accuracy: {(total_correct/total_words*100):.2f}%")
    print(f"{'='*80}\n")
    
    # Print detailed results
    print(f"{'Index':<6} {'Correct':<10} {'Total':<10} {'Accuracy':<12} {'Reference':<35} {'Hypothesis':<35}")
    print(f"{'-'*130}")
    
    for result in sample_results:
        ref_short = result['reference'][:33] + '..' if len(result['reference']) > 35 else result['reference']
        hyp_short = result['hypothesis'][:33] + '..' if len(result['hypothesis']) > 35 else result['hypothesis']
        print(f"{result['index']:<6} {result['correct']:<10} {result['total']:<10} {result['accuracy']:<12} {ref_short:<35} {hyp_short:<35}")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("STATISTICS:")
    print(f"{'='*80}")
    
    correct_samples = sum(1 for r in sample_results if r['correct'] == r['total'])
    partially_correct = sum(1 for r in sample_results if 0 < r['correct'] < r['total'])
    incorrect_samples = sum(1 for r in sample_results if r['correct'] == 0)
    
    print(f"Fully correct samples: {correct_samples} ({correct_samples/len(per_sample)*100:.1f}%)")
    print(f"Partially correct samples: {partially_correct} ({partially_correct/len(per_sample)*100:.1f}%)")
    print(f"Incorrect samples: {incorrect_samples} ({incorrect_samples/len(per_sample)*100:.1f}%)")

if __name__ == '__main__':
    main()
