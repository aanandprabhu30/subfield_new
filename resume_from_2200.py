#!/usr/bin/env python3
"""
Resume classification from extracted checkpoint at 2200 papers
"""

import os
import sys
import shutil
from pathlib import Path

def setup_resume_from_2200():
    """Setup files for resuming classification from 2200 papers"""
    
    # Check if extracted checkpoint exists
    if not Path('checkpoint_advanced_current_2200.pkl').exists():
        print("❌ checkpoint_advanced_current_2200.pkl not found!")
        return False
    
    # Rename the extracted checkpoint to the expected name
    shutil.copy('checkpoint_advanced_current_2200.pkl', 'checkpoint_advanced.pkl')
    print("✅ Copied checkpoint_advanced_current_2200.pkl to checkpoint_advanced.pkl")
    
    # Also copy the extracted CSV as a backup
    if Path('classified_papers_hybrid_final_current_2200.csv').exists():
        shutil.copy('classified_papers_hybrid_final_current_2200.csv', 'classified_papers_hybrid_final_partial_2200.csv')
        print("✅ Copied extracted progress to classified_papers_hybrid_final_partial_2200.csv")
    
    return True

def main():
    """Main function to resume classification from 2200 papers"""
    
    print("🔄 Setting up resume from extracted checkpoint at 2200 papers...")
    
    if not setup_resume_from_2200():
        print("❌ Failed to setup resume!")
        return
    
    print("\n✅ Ready to resume classification!")
    print("📊 Progress: 2200/26,944 papers (8.2%)")
    print("💰 Cost so far: ~$1.65")
    print("\n🚀 Starting classification with FIXED checkpoint saving...")
    print("📝 Checkpoints will now be saved every 100 papers (backup) and 500 papers (main)")
    
    # Run the classifier with resume flag
    cmd = "conda run python abstract_classifier_advanced.py --input Abstracts.csv --output classified_papers_hybrid_final.csv --resume"
    print(f"\nRunning: {cmd}")
    
    # Execute the command
    os.system(cmd)

if __name__ == "__main__":
    main() 