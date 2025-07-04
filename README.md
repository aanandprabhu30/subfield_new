# Research Paper Abstract Classification System

A high-performance Python system for classifying research paper abstracts into computing disciplines and subfields using OpenAI's GPT models.

## Features

- **Parallel Processing**: Up to 100 concurrent API calls for maximum throughput
- **Smart Pre-filtering**: Reduces API calls by identifying non-computing papers
- **Two-Stage Classification**: Primary model (GPT-4o-mini) with fallback to GPT-4o for difficult cases
- **Cost Optimization**: Achieves <$0.10 cost for full dataset using GPT-4o-mini
- **Robust Error Handling**: Automatic retries with exponential backoff
- **Checkpoint System**: Resume processing from any interruption
- **Comprehensive Caching**: LRU cache for processed papers
- **Detailed Statistics**: Track costs, confidence scores, and distributions

## Installation

**Clone the repository**:

```bash
git clone <repository>
cd subfield_new
```

**Install dependencies** (uses only built-in libraries):

```bash
# No external dependencies required - uses built-in Python libraries only
# Requires Python 3.8+
```

**Set up API key**:

```bash
export OPENAI_API_KEY="sk-proj-your-api-key-here"
# Or create a .env file:
echo "OPENAI_API_KEY=sk-proj-your-api-key-here" > .env
```

## Usage

### Basic Usage

```bash
python classify_abstracts.py --input papers.csv --output results.csv
```

### Resume from Checkpoint

```bash
python classify_abstracts.py --resume --input papers.csv --output results.csv
```

### Command-Line Options

- `--input`: Input CSV file (default: Abstracts.csv)
- `--output`: Output CSV file (default: classified_papers.csv)
- `--resume`: Resume from checkpoint if available
- `--api-key`: OpenAI API key (alternative to environment variable)

## Input Format

CSV file with the following columns:

```csv
Title,Abstract
"Deep Learning for Computer Vision","This paper presents..."
"Enterprise Resource Planning","This study examines..."
```

## Output Files

1. **Main Results** (`output.csv`):
   - All original columns
   - `Primary Discipline`: CS/IS/IT/NON_COMPUTING
   - `Subfield`: Specific subfield name
   - `Confidence`: 0-100 confidence score
   - `Model Used`: gpt-4o-mini/gpt-4o
   - `Reasoning`: Classification explanation
   - `Key Indicators`: Supporting evidence
   - `Alternative`: Second-best classification
   - `Cached`: Whether result was cached

2. **Review File** (`output_review.csv`):
   - Papers with confidence < 70% for manual review

3. **Error File** (`output_errors.csv`):
   - Failed classifications for reprocessing

4. **Statistics** (`output_stats.json`):
   - Processing statistics and distributions

## Classification Schema

### Primary Disciplines

1. **Computer Science (CS)**: Theoretical algorithms, software development, technical research
2. **Information Systems (IS)**: Business applications, organizational technology, enterprise systems
3. **Information Technology (IT)**: Practical implementation, infrastructure, operations

### Subfields

Each discipline has 25 specific subfields. Examples:

- **CS**: AI & ML, Computer Vision, NLP, Robotics, Data Science, etc.
- **IS**: Enterprise Systems, Business Intelligence, E-commerce, etc.
- **IT**: Infrastructure, DevOps, Cloud Management, IT Security, etc.

## Performance

### Expected Metrics

- **Classification Rate**: >90% successful
- **UNKNOWN Rate**: <10% of computing papers
- **Confidence Scores**: >70% average
- **Processing Speed**: ~300 papers/minute
- **Cost**: <$0.10 for 18,251 papers

### Optimization Features

- Pre-filtering reduces unnecessary API calls
- Dynamic batch sizing adapts to API performance
- Concurrent processing maximizes throughput
- Caching prevents duplicate processing

## Cost Estimation

Based on GPT-4o-mini pricing:

- Input: $0.00015 per 1K tokens
- Output: $0.0006 per 1K tokens

For 18,251 papers:

- Estimated tokens: ~7M input, 1.5M output
- Estimated cost: ~$0.08-0.10 total

## Troubleshooting

### Issue: Processing shows 0 papers

**Solution**: Check if papers are already in checkpoint cache. Delete `checkpoint.pkl` to reprocess.

### Issue: High UNKNOWN rate

**Solution**:

- Review computing keywords in pre-filter
- Check if abstracts contain technical terms
- Consider adjusting confidence threshold

### Issue: API rate limits

**Solution**:

- Reduce `MAX_CONCURRENT_CALLS` in script
- Adjust `MAX_REQUESTS_PER_MINUTE`
- Script automatically handles rate limits with backoff

### Issue: Low confidence scores

**Solution**:

- Papers may be interdisciplinary
- Review `output_review.csv` for patterns
- Consider manual classification for edge cases

## Architecture

### Pre-filtering Function

```python
def is_computing_related(title: str, abstract: str) -> bool:
    """Identifies computing-related papers using keyword matching"""
    # Requires at least 2 computing keywords
```

### Two-Stage Classification

1. **Primary Classification** (GPT-4o-mini):
   - Fast, cost-effective classification
   - Handles majority of papers

2. **Fallback Classification** (GPT-4o):
   - Triggered when confidence < 70%
   - More detailed analysis
   - Higher accuracy for difficult cases

### Parallel Processing

- Async/await for concurrent API calls
- Semaphore limits to 100 concurrent requests
- Rate limiter prevents API throttling

## Monitoring Progress

The script provides real-time updates:

``` text
Processing batch 5/183
Progress: 500/18251 (2.7%), Rate: 285/min, ETA: 62.3min, Cost: $0.04
```

Log file (`classification.log`) contains detailed information for debugging.

## Best Practices

1. **Start Small**: Test with a subset of papers first
2. **Monitor Costs**: Check statistics regularly
3. **Review Low Confidence**: Papers in `_review.csv` may need manual verification
4. **Use Checkpoints**: Enable resuming for large datasets
5. **Validate Results**: Spot-check classifications for accuracy

## Support

For issues:

1. Check the log file for detailed errors
2. Ensure API key has sufficient credits
3. Verify CSV format matches expected structure
4. Review checkpoint file if resuming

## License

[Your License Here]
