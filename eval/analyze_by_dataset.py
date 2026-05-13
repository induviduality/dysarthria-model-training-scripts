import json

# Mapping of speakers to their datasets based on training data
SPEAKER_TO_DATASET = {
    # TORGO dataset speakers
    'F03': 'TORGO',
    'F04': 'TORGO',
    'M02': 'TORGO',
    'M03': 'TORGO',
    'M04': 'TORGO',
    'M05': 'TORGO',
    
    # UASpeech dataset speakers
    'F02': 'UASpeech',
    'F05': 'UASpeech',
    'M01': 'UASpeech',
    'M08': 'UASpeech',
    'M09': 'UASpeech',
    'M10': 'UASpeech',
    'M11': 'UASpeech',
    'M12': 'UASpeech',
    'M14': 'UASpeech',
    'M16': 'UASpeech',
}

def analyze_model(filepath, model_name):
    """Analyze a single model's results by dataset"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except:
        print(f"Could not load {filepath}")
        return
    
    per_speaker_wer = data.get('per_speaker_wer', {})
    overall_wer = data.get('overall_wer', 0)
    overall_cer = data.get('overall_cer', 0)
    
    # Group speakers by dataset
    dataset_stats = {}
    
    for speaker_id, metrics in per_speaker_wer.items():
        dataset = SPEAKER_TO_DATASET.get(speaker_id, 'Unknown')
        
        if dataset not in dataset_stats:
            dataset_stats[dataset] = {
                'wer': [],
                'cer': [],
                'n_samples': 0,
                'speakers': []
            }
        
        dataset_stats[dataset]['wer'].append(metrics['wer'])
        dataset_stats[dataset]['cer'].append(metrics['cer'])
        dataset_stats[dataset]['n_samples'] += metrics['n_samples']
        dataset_stats[dataset]['speakers'].append(speaker_id)
    
    # Calculate average WER/CER by dataset
    print(f"\n{'='*100}")
    print(f"EVALUATION RESULTS BY DATASET SOURCE - {model_name}")
    print(f"{'='*100}\n")
    
    print(f"Overall Model Performance:")
    print(f"  Overall WER: {overall_wer:.3f}%")
    print(f"  Overall CER: {overall_cer:.3f}%")
    print(f"  Total Samples: {data.get('n_samples', 'N/A')}")
    print(f"  Model: {data.get('model_path', 'N/A')}")
    print(f"  Split: {data.get('split', 'N/A')}")
    
    print(f"\n{'='*100}\n")
    print(f"{'Dataset':<15} {'Avg WER':<12} {'Avg CER':<12} {'Speakers':<20} {'Total Samples':<15}")
    print(f"{'-'*100}")
    
    dataset_wer_map = {}
    
    for dataset in sorted(dataset_stats.keys()):
        stats = dataset_stats[dataset]
        avg_wer = sum(stats['wer']) / len(stats['wer']) if stats['wer'] else 0
        avg_cer = sum(stats['cer']) / len(stats['cer']) if stats['cer'] else 0
        
        dataset_wer_map[dataset] = avg_wer
        
        print(f"{dataset:<15} {avg_wer:>10.3f}% {avg_cer:>10.3f}% {len(stats['speakers']):>6} speakers   {stats['n_samples']:>6} samples")
    
    print(f"\n{'='*100}\n")
    print("PER-SPEAKER WER/CER BREAKDOWN:")
    print(f"{'='*100}\n")
    
    print(f"{'Speaker':<10} {'Dataset':<15} {'WER':<12} {'CER':<12} {'Samples':<10}")
    print(f"{'-'*100}")
    
    for speaker_id in sorted(per_speaker_wer.keys()):
        metrics = per_speaker_wer[speaker_id]
        dataset = SPEAKER_TO_DATASET.get(speaker_id, 'Unknown')
        print(f"{speaker_id:<10} {dataset:<15} {metrics['wer']:>10.3f}% {metrics['cer']:>10.3f}% {metrics['n_samples']:>6}")
    
    print(f"\n{'='*100}\n")
    print("DATASET RANKING (by WER - lower is better):")
    print(f"{'='*100}\n")
    
    for idx, (dataset, wer) in enumerate(sorted(dataset_wer_map.items(), key=lambda x: x[1]), 1):
        print(f"{idx}. {dataset:<15} - Avg WER: {wer:>8.3f}%")
    
    return data, dataset_stats

def main():
    print("\n" + "="*100)
    print("VALIDATION SET ANALYSIS - BY DATASET SOURCE")
    print("="*100)
    
    # Analyze base model
    base_data, base_stats = analyze_model('inference-outputs/evaluation_results_base.json', 'BASE MODEL')

    # Analyze finetuned model
    tuned_data, tuned_stats = analyze_model('inference-outputs/evaluation_results.json', 'FINETUNED MODEL')
    
    # Comparison
    if base_data and tuned_data:
        print(f"\n{'='*100}")
        print("IMPROVEMENT COMPARISON (Finetuned vs Base)")
        print(f"{'='*100}\n")
        
        print(f"{'Dataset':<15} {'Base WER':<12} {'Tuned WER':<12} {'Improvement':<15} {'Direction':<15}")
        print(f"{'-'*100}")
        
        for dataset in sorted(set(base_stats.keys()) | set(tuned_stats.keys())):
            if dataset in base_stats and dataset in tuned_stats:
                base_wer = sum(base_stats[dataset]['wer']) / len(base_stats[dataset]['wer'])
                tuned_wer = sum(tuned_stats[dataset]['wer']) / len(tuned_stats[dataset]['wer'])
                improvement = base_wer - tuned_wer
                direction = "✓ Better" if improvement > 0 else "✗ Worse" if improvement < 0 else "="
                
                print(f"{dataset:<15} {base_wer:>10.3f}% {tuned_wer:>10.3f}% {improvement:>+8.3f}pp    {direction:<15}")

if __name__ == '__main__':
    main()
