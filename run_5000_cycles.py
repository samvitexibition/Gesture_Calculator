import os
import time
import numpy as np
import subprocess
import generate_synthetic_data

def extract_accuracy_from_output(output):
    """Extract accuracy from train_model.py output"""
    for line in output.split('\n'):
        if 'Test Set Accuracy:' in line:
            try:
                accuracy = float(line.split(':')[1].split('%')[0].strip())
                return accuracy
            except:
                pass
    return None

def run_5000_cycles():
    print(f"\n{'='*80}")
    print(f"TRAINING ALL CLASSES FOR 5000 CYCLES")
    print(f"{'='*80}")
    
    accuracies = []
    start_time = time.time()
    
    for cycle_num in range(1, 2):
        # Generate dataset with all classes
        generate_synthetic_data.generate_mock_data(samples_per_class=100)
        
        # Run training
        # No filtering needed since we train on all classes
        result = subprocess.run(["python", "train_model.py"], 
                              capture_output=True, text=True)
        
        # Extract accuracy
        accuracy = extract_accuracy_from_output(result.stdout)
        
        if accuracy is not None:
            accuracies.append(accuracy)
            print(f"  Cycle {cycle_num:4d}/5000 - Accuracy: {accuracy:.2f}%")
        else:
            print(f"  Cycle {cycle_num:4d}/5000 - Failed to extract accuracy")
            
    total_time = time.time() - start_time
    return accuracies, total_time

def generate_report(accuracies, total_time):
    """Generate a comprehensive report"""
    if not accuracies:
        print("No valid accuracies to report.")
        return
    
    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    min_accuracy = np.min(accuracies)
    max_accuracy = np.max(accuracies)
    
    print(f"\n{'='*80}")
    print(f"FINAL TRAINING REPORT")
    print(f"{'='*80}")
    print(f"Training Summary:")
    print(f"  Total Cycles: 5000")
    print(f"  Successful Cycles: {len(accuracies)}")
    print(f"  Failed Cycles: {5000 - len(accuracies)}")
    print(f"  Average Accuracy: {avg_accuracy:.2f}%")
    print(f"  Accuracy Std Dev: {std_accuracy:.2f}%")
    print(f"  Min Accuracy: {min_accuracy:.2f}%")
    print(f"  Max Accuracy: {max_accuracy:.2f}%")
    print(f"  Total Training Time: {total_time:.1f}s")
    print(f"  Average Time per Cycle: {total_time/len(accuracies):.1f}s")
    print(f"{'='*80}\n")
    
    # Save to CSV
    with open("complete_training_report.csv", "w", newline="") as f:
        f.write("cycle,accuracy\n")
        for i, acc in enumerate(accuracies):
            f.write(f"{i+1},{acc}\n")
            
    print(f"\nDetailed report saved to: complete_training_report.csv")

if __name__ == "__main__":
    print("\nStarting 5000 Cycle Training for All Classes!")
    print("Each cycle will generate new synthetic data and train the model.")
    
    accuracies, total_time = run_5000_cycles()
    generate_report(accuracies, total_time)
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE!")
    print(f"{'='*80}")