#!/usr/bin/env python3
"""
Test script for the advanced abstract classifier
"""

import asyncio
import csv
import os
from abstract_classifier_advanced import AbstractClassifier, RateLimiter

# Test abstracts covering different disciplines
TEST_ABSTRACTS = [
    {
        "Title": "Deep Learning Approaches for Medical Image Segmentation",
        "Abstract": "This paper presents a novel deep learning architecture for automatic segmentation of medical images. We propose a modified U-Net architecture with attention mechanisms that achieves state-of-the-art performance on brain tumor segmentation tasks. Our model uses convolutional neural networks with skip connections and incorporates recent advances in self-attention layers. Experiments on the BraTS dataset show significant improvements over baseline methods, achieving a Dice coefficient of 0.92. The proposed method has potential applications in clinical diagnosis and treatment planning.",
        "Expected_Discipline": "CS",
        "Expected_Subfield": "AI_ML"
    },
    {
        "Title": "Digital Transformation Strategy for Manufacturing Enterprises",
        "Abstract": "This study examines how manufacturing companies can develop and implement effective digital transformation strategies. Through case studies of five large enterprises, we identify key success factors including leadership commitment, organizational culture change, and strategic technology investments. The research reveals that successful digital transformation requires alignment between IT strategy and business objectives, with emphasis on change management and employee training. We propose a maturity model for assessing digital transformation readiness and provide recommendations for implementation roadmaps.",
        "Expected_Discipline": "IS",
        "Expected_Subfield": "DT"
    },
    {
        "Title": "Implementing DevOps Practices in Cloud Infrastructure",
        "Abstract": "This paper describes practical approaches to implementing DevOps methodologies in cloud-based environments. We present a comprehensive guide covering CI/CD pipeline setup, infrastructure as code using Terraform, container orchestration with Kubernetes, and monitoring strategies. Our case study demonstrates how a financial services company reduced deployment time by 80% and improved system reliability through automated testing and deployment processes. Key topics include GitOps workflows, security scanning integration, and performance optimization techniques for AWS and Azure environments.",
        "Expected_Discipline": "IT",
        "Expected_Subfield": "DEVOPS"
    },
    {
        "Title": "Quantum Algorithms for Cryptographic Applications",
        "Abstract": "We present new quantum algorithms for solving cryptographic problems that are believed to be hard for classical computers. Our work focuses on lattice-based cryptography and develops quantum algorithms that achieve polynomial speedup over classical approaches. We analyze the complexity of our algorithms and discuss implications for post-quantum cryptography. Theoretical analysis shows that our methods could potentially break certain lattice-based encryption schemes with sufficient quantum resources. We also propose modifications to existing cryptographic protocols to resist these quantum attacks.",
        "Expected_Discipline": "CS",
        "Expected_Subfield": "QUANTUM"
    },
    {
        "Title": "Healthcare Analytics Platform for Patient Outcome Prediction",
        "Abstract": "This research presents an integrated healthcare analytics platform designed to predict patient outcomes and support clinical decision-making. The system leverages electronic health records, real-time monitoring data, and machine learning models to identify high-risk patients and recommend interventions. We implemented the platform in three hospitals and demonstrated a 25% reduction in readmission rates. The platform includes dashboards for clinicians, predictive models for various conditions, and integration with existing hospital information systems. Key features include HIPAA compliance, real-time alerts, and explainable AI components.",
        "Expected_Discipline": "IS",
        "Expected_Subfield": "HIS"
    }
]

async def test_classifier():
    """Test the classifier with sample abstracts"""
    print("=== Testing Advanced Abstract Classifier ===\n")
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Please set OPENAI_API_KEY environment variable")
        return
    
    # Initialize classifier
    rate_limiter = RateLimiter(100)  # Lower rate limit for testing
    
    async with AbstractClassifier(api_key, rate_limiter) as classifier:
        results = []
        
        for i, test_case in enumerate(TEST_ABSTRACTS):
            print(f"Test {i+1}/{len(TEST_ABSTRACTS)}: {test_case['Title'][:50]}...")
            
            # Create paper dict
            paper = {
                "Title": test_case["Title"],
                "Abstract": test_case["Abstract"]
            }
            
            # Classify
            result = await classifier.classify_abstract(paper)
            
            # Check results
            discipline_match = result['Discipline'] == test_case['Expected_Discipline']
            subfield_match = result['Subfield'] == test_case['Expected_Subfield']
            
            print(f"  Expected: {test_case['Expected_Discipline']}/{test_case['Expected_Subfield']}")
            print(f"  Got: {result['Discipline']}/{result['Subfield']}")
            print(f"  Confidence: {result['Confidence_Score']:.2f}")
            print(f"  Match: {'✓' if discipline_match and subfield_match else '✗'}")
            print(f"  Reasoning: {result.get('Reasoning', 'N/A')[:100]}...")
            print()
            
            results.append({
                'title': test_case['Title'],
                'expected_discipline': test_case['Expected_Discipline'],
                'expected_subfield': test_case['Expected_Subfield'],
                'actual_discipline': result['Discipline'],
                'actual_subfield': result['Subfield'],
                'confidence': result['Confidence_Score'],
                'discipline_match': discipline_match,
                'subfield_match': subfield_match,
                'reasoning': result.get('Reasoning', '')
            })
        
        # Summary
        print("=== Test Summary ===")
        total_tests = len(results)
        discipline_correct = sum(1 for r in results if r['discipline_match'])
        subfield_correct = sum(1 for r in results if r['subfield_match'])
        both_correct = sum(1 for r in results if r['discipline_match'] and r['subfield_match'])
        avg_confidence = sum(r['confidence'] for r in results) / total_tests
        
        print(f"Total tests: {total_tests}")
        print(f"Discipline accuracy: {discipline_correct}/{total_tests} ({discipline_correct/total_tests*100:.1f}%)")
        print(f"Subfield accuracy: {subfield_correct}/{total_tests} ({subfield_correct/total_tests*100:.1f}%)")
        print(f"Both correct: {both_correct}/{total_tests} ({both_correct/total_tests*100:.1f}%)")
        print(f"Average confidence: {avg_confidence:.2f}")
        
        # Cost estimate
        print(f"\nEstimated cost: ${classifier.get_total_cost():.4f}")
        
        # Save detailed results
        with open('test_results.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print("\nDetailed results saved to test_results.csv")


if __name__ == "__main__":
    asyncio.run(test_classifier())