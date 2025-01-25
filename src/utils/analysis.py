import json
import argparse
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
# Model name mappings
MODEL_NAMES = {
    'deepseek_r1': 'DeepSeek R1',
    'claude_standalone': 'Sonnet 20241022',
    'claude_with_reasoning': 'Sonnet 200241022+R1R'
}

def analyze_results(jsonl_path):
    """Main analysis function that processes the JSONL file"""
    results = {
        'overall': {
            'total_questions': 0,
            'models': {
                'deepseek_r1': {'correct': 0},
                'claude_standalone': {'correct': 0},
                'claude_with_reasoning': {'correct': 0},
            }
        },
        'difficulty': defaultdict(lambda: {
            'total': 0,
            'models': {
                'deepseek_r1': {'correct': 0},
                'claude_standalone': {'correct': 0},
                'claude_with_reasoning': {'correct': 0},
            }
        }),
        'domains': defaultdict(lambda: {
            'total': 0,
            'models': {
                'deepseek_r1': {'correct': 0},
                'claude_standalone': {'correct': 0},
                'claude_with_reasoning': {'correct': 0},
            }
        }),
        'errors': defaultdict(list)
    }

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line.strip())
                process_entry(entry, results, line_num)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error processing line {line_num}: {e}")
                continue

    calculate_derived_metrics(results)
    return results

def process_entry(entry, results, line_num):
    """Process a single JSONL entry with null safety"""
    results['overall']['total_questions'] += 1
    
    # Get raw difficulty with fallback to empty string
    raw_difficulty = entry['metadata'].get('difficulty', '')
    difficulty = map_difficulty(raw_difficulty)
    
    # Handle domain consistently
    domain = entry['metadata'].get('high_level_domain', 'Unknown')
    
    results['difficulty'][difficulty]['total'] += 1
    results['domains'][domain]['total'] += 1

    models = {
        'deepseek_r1': 'deepseek_reasoner_grade',
        'claude_standalone': 'claude_sonnet_standalone_grade',
        'claude_with_reasoning': 'claude_sonnet_with_reasoning_grade'
    }

    for model, grade_key in models.items():
        is_correct = entry.get(grade_key, False)
        if is_correct:
            results['overall']['models'][model]['correct'] += 1
            results['difficulty'][difficulty]['models'][model]['correct'] += 1
            results['domains'][domain]['models'][model]['correct'] += 1
        else:
            results['errors'][model].append({
                'question': entry['question'],
                'correct': entry['correct_answer'],
                'model': entry.get(f"{grade_key.replace('_grade', '')}_answer", "N/A")
            })

def calculate_derived_metrics(results):
    """Calculate percentages for all metrics"""
    total_questions = results['overall']['total_questions']
    
    for model in results['overall']['models']:
        correct = results['overall']['models'][model]['correct']
        results['overall']['models'][model]['percentage'] = safe_divide(correct, total_questions) * 100

    for difficulty in results['difficulty']:
        total = results['difficulty'][difficulty]['total']
        for model in results['difficulty'][difficulty]['models']:
            correct = results['difficulty'][difficulty]['models'][model]['correct']
            results['difficulty'][difficulty]['models'][model]['percentage'] = safe_divide(correct, total) * 100

    for domain in results['domains']:
        total = results['domains'][domain]['total']
        for model in results['domains'][domain]['models']:
            correct = results['domains'][domain]['models'][model]['correct']
            results['domains'][domain]['models'][model]['percentage'] = safe_divide(correct, total) * 100

def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0

def create_plots(results, output_dir):
    """Generate all visualizations"""
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.figsize': (12, 6),
        'figure.dpi': 300,
        'axes.titlepad': 20
    })
    
    create_win_rate_chart(results, output_dir)
    create_difficulty_chart(results, output_dir)
    create_domain_chart(results, output_dir)
    create_error_chart(results, output_dir)
    create_radar_chart(results, output_dir)

def create_win_rate_chart(results, output_dir):
    """Adjusted bar chart with narrower columns and enhanced contrast"""
    accuracies = [
        results['overall']['models']['deepseek_r1']['percentage'],
        results['overall']['models']['claude_standalone']['percentage'],
        results['overall']['models']['claude_with_reasoning']['percentage']
    ]
    
    plt.figure(figsize=(9, 6))  # Taller aspect ratio
    ax = plt.gca()
    
    # Create positions and narrower bars
    x_pos = np.arange(len(MODEL_NAMES))
    bar_width = 0.45  # Reduced from default 0.8
    
    bars = ax.bar(x_pos, accuracies, width=bar_width,
                color=sns.color_palette("rocket", len(MODEL_NAMES)),
                edgecolor='white',
                linewidth=1.5,
                alpha=0.95)
    
    # Enhanced annotations
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                   (bar.get_x() + bar.get_width()/2, height),
                   ha='center', va='bottom',
                   xytext=(0, 8),
                   textcoords='offset points',
                   fontsize=13,
                   weight='bold',
                   color='#2d2d2d')

    # Improved y-axis scaling
    max_acc = max(accuracies)
    min_acc = min(accuracies)
    y_buffer = (max_acc - min_acc) * 0.15  # 15% of difference
    ax.set_ylim(max(0, min_acc - y_buffer), min(100, max_acc + y_buffer))
    
    # Final styling
    ax.set_title('Overall Model Accuracy Comparison', pad=20, fontweight='semibold')
    ax.set_ylabel('Accuracy (%)', labelpad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(MODEL_NAMES.values(), rotation=12, ha='right')
    ax.yaxis.grid(True, alpha=0.3)
    sns.despine(left=True, bottom=True)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'win_rate_chart.png', bbox_inches='tight', dpi=300)
    plt.close()

def map_difficulty(original):
    """Map raw difficulty string to standardized categories"""
    if not original:
        return 'Unknown'
    
    original_lower = original.lower()
    
    # Check specific patterns first
    if re.search(r'post-graduate level or harder', original_lower):
        return 'Postgrad'
    if re.search(r'hard graduate level', original_lower):
        return 'Graduate'
    if re.search(r'hard undergraduate level', original_lower):
        return 'Hard Undergrad'
    if re.search(r'easy undergraduate level', original_lower):
        return 'Easy Undergrad'
    
    # General keyword fallbacks
    if re.search(r'\b(hard|advanced|challenging)\b', original_lower):
        return 'Hard'
    if re.search(r'\b(easy|introductory|beginner)\b', original_lower):
        return 'Easy'
    
    # Academic level fallbacks
    if re.search(r'\b(postgrad|post-grad|postgraduate)\b', original_lower):
        return 'Postgrad'
    if re.search(r'\bgrad(?!.*under)\b', original_lower):
        return 'Graduate'
    if re.search(r'\b(undergrad|undergraduate|bachelor)\b', original_lower):
        return 'Undergrad'
    
    return 'Unknown'

def create_difficulty_chart(results, output_dir):
    """Improved difficulty chart with consistent ordering"""
    DIFFICULTY_ORDER = ['Postgrad', 'Graduate', 'Hard Undergrad', 'Easy Undergrad', 'Unknown']
    present_difficulties = [d for d in DIFFICULTY_ORDER if results['difficulty'][d]['total'] > 0]
    
    if not present_difficulties:
        return

    plt.figure(figsize=(12, 7))
    ax = plt.gca()
    
    n_models = len(MODEL_NAMES)
    bar_width = 0.25
    x = np.arange(len(present_difficulties))
    
    colors = sns.color_palette("husl", n_models)
    
    for idx, model in enumerate(MODEL_NAMES.keys()):
        percentages = [results['difficulty'][d]['models'][model]['percentage'] 
                      for d in present_difficulties]
        
        offset = bar_width * idx
        bars = ax.bar(x + offset, percentages, bar_width,
                     label=MODEL_NAMES[model], color=colors[idx],
                     edgecolor='white', linewidth=0.7)
        
        for bar in bars:
            height = bar.get_height()
            if height > 3:
                ax.annotate(f'{height:.1f}%',
                           (bar.get_x() + bar.get_width()/2, height),
                           ha='center', va='bottom',
                           fontsize=10, color='black')

    ax.set_title('Model Performance by Difficulty Level', pad=15)
    ax.set_ylabel('Accuracy (%)', labelpad=12)
    ax.set_ylim(0, 100)
    
    ax.set_xticks(x + bar_width*(n_models-1)/2)
    ax.set_xticklabels(present_difficulties, rotation=25, 
                      ha='right', rotation_mode='anchor')
    
    ax.legend(title='Models', frameon=True, 
             bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'difficulty_chart.png', bbox_inches='tight')
    plt.close()

def create_domain_chart(results, output_dir):
    """Heatmap of domain performance"""
    domains = sorted(results['domains'].keys())
    
    data = np.array([[results['domains'][d]['models'][m]['percentage'] 
                    for d in domains] for m in MODEL_NAMES.keys()])
    
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(data, annot=True, fmt=".1f", cmap="YlGnBu",
               xticklabels=domains, yticklabels=MODEL_NAMES.values(),
               cbar_kws={'label': 'Accuracy (%)'})
    
    ax.set_title('Model Accuracy Across Domains', pad=20)
    ax.set_xlabel('Domain', labelpad=15)
    ax.set_ylabel('Model', labelpad=15)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'domain_heatmap.png')
    plt.close()

def create_error_chart(results, output_dir):
    """Error distribution plot"""
    error_counts = [len(results['errors'][m]) for m in MODEL_NAMES.keys()]
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(list(MODEL_NAMES.values()), error_counts,
                   color=sns.color_palette("Reds_r", 3))
    
    plt.title('Error Distribution Across Models', pad=20)
    plt.xlabel('Number of Errors', labelpad=15)
    
    # Add accuracy annotations
    for i, model in enumerate(MODEL_NAMES.keys()):
        accuracy = results['overall']['models'][model]['percentage']
        plt.text(error_counts[i]+5, i, 
                f'{accuracy:.1f}% Accuracy', 
                va='center', color='#2e2e2e')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'error_distribution.png')
    plt.close()

def create_radar_chart(results, output_dir):
    """Radar chart comparing model performance"""
    categories = list(results['domains'].keys())
    if not categories:
        return

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    
    for model in MODEL_NAMES.keys():
        values = [results['domains'][d]['models'][model]['percentage'] for d in categories]
        values += values[:1]  # Close the polygon
        ax.plot(angles + angles[:1], values, marker='o', linestyle='-', 
               linewidth=2, markersize=8, label=MODEL_NAMES[model])
    
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles), categories)
    plt.yticks([25, 50, 75], ["25%", "50%", "75%"], color="grey", size=10)
    plt.ylim(0, 100)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title('Model Performance Across Domains', pad=40)
    plt.savefig(Path(output_dir) / 'radar_chart.png', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze model performance from JSONL file')
    parser.add_argument('input_file', type=str, help='Path to input JSONL file')
    parser.add_argument('--output-dir', type=str, default='./analysis_results',
                      help='Output directory for charts (default: ./analysis_results)')
    
    args = parser.parse_args()
    
    if not Path(args.input_file).exists():
        print(f"Error: Input file {args.input_file} not found")
        exit(1)
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    analysis_results = analyze_results(args.input_file)
    create_plots(analysis_results, args.output_dir)
    
    print(f"Analysis complete! Charts saved to {args.output_dir}:")
    print("- win_rate_chart.png\n- difficulty_chart.png\n- domain_heatmap.png")
    print("- error_distribution.png\n- radar_chart.png")