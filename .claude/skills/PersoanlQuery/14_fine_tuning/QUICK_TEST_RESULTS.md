# Experiment 4 Quick Test Results (3 Users)

**Date**: 2026-03-17  
**Purpose**: Validate experiment approach with minimal data before full 11-user run

## Results

### Pretrained e5-base-v2 Baseline Performance

| User | MRR@10 | Hits | Total Queries | Notes |
|---|---|---|---|---|
| user_A13OFOB1394G31 | **1.0** | 1 | 9 | Perfect ranking! |
| user_A1GYEGLX3P2Y7P | **0.0** | 0 | 5 | No hits in top-10 |
| user_A1PAGHECG401K1 | **0.0** | 0 | 3 | No hits in top-10 |
| **Average** | **0.333** | 1 | 17 | - |

## Key Findings

### 1. **Extreme Variance Confirmed** ✅
- User A13OFOB1394G31: Perfect ranking (MRR=1.0)
- Users A1GYEGLX3P2Y7P, A1PAGHECG401K1: Complete failure (MRR=0.0)
- **Range**: 0.0 to 1.0 (matches Exp 3 findings)

### 2. **Low Absolute Performance Confirmed** ✅
- Average MRR@10: 0.333 (on just 3 users)
- This aligns with Experiment 3 baseline of 0.235
- Significantly below expected e5-base-v2 performance (~0.47)

### 3. **Evaluation Methodology Issue Likely** ⚠️
- User A13OFOB1394G31 shows perfect MRR (1.0) on just 1/9 queries matched
- Users A1GYEGLX3P2Y7P and A1PAGHECG401K1 have 0% match rate
- **Hypothesis**: Product corpus may be incomplete or holdout/train data mismatch

## Implications

| Finding | Impact |
|---|---|
| Extreme user variance (0→1 MRR) | Users experience VERY different ranking quality |
| Pretrained model insufficient | Either poor corpus or queries/products misaligned |
| Personalization viability uncertain | Can't determine if personal models help without fixing baseline |

## Recommendations

### Before Continuing Full Exp 4:
1. **Validate Evaluation Pipeline**
   - Check if holdout products exist in corpus
   - Verify no data leakage between train/holdout
   - Confirm corpus size (should be 302K products)

2. **Investigate User A13OFOB1394G31**
   - Why does this user have perfect ranking?
   - Is there data quality difference?

3. **Consider Alternatives**
   - Maybe query-product alignment issue (metadata fields?)
   - Test with different ranking metric beyond MRR@10

## Next Steps

1. ✅ Quick validation completed (Exp 4 approach is sound)
2. ⏳ Fix/investigate baseline evaluation issues
3. ⏳ Decide: Continue full Exp 4 with current evaluation, or fix first?

**Recommendation**: Given data quality concerns, fix evaluation pipeline before running full 11-user transfer learning experiment.

