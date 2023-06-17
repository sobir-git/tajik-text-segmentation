'''
Sentence Boundary Resolver

This algorithm takes the predictions of a neural network, which indicate the likelihood 
of each token being the start or end of a sentence. It addresses the issue of inconsistent 
predictions by refining and producing actual valid sentence boundaries.

By analyzing the predictions at the token level and considering the overall coherence of 
the sentence, the Sentence Boundary Resolver algorithm harmonizes and adjusts the predicted 
boundaries. It ensures that the generated sentence boundaries accurately represent natural 
language sentences.

This algorithm is particularly useful in tasks involving text segmentation, document processing, 
and natural language understanding, where identifying correct sentence boundaries is essential 
for downstream analysis and applications.
'''

import numpy as np


class SentenceBoundaryResolver:
    def __init__(self, max_candidates=10, candidate_threshold=0.25) -> None:
        self.max_candidates = max_candidates
        self.candidate_threshold = candidate_threshold

    def resolve(self, probs: np.ndarray, binarize_output=False):
        '''
        Resolve sentence boundaries by looking at individual token "start" and "end" probalities.
        
        Argument:
            probs: array of shape (N, 2) where N represents sequence length and 
                     2 represents the entries for sentence start and sentence end
                     probabilities
        Return:
            boundaries: - if binarize_output is True, then returns a list of (start_index, end_index) 
                        sentence boundaries where start_index < end_index where start_index is included 
                        and end_index is excluded
                        - else, returns a binary array of same shape as probs with the same semantics
        '''

        # The algorithm tries to maximize the segmentation likelyhood defined as the sum of log probabilities
        # of individual choices (is_start, not_is_start, is_end, not_is_end) made at each token.
        # Note that not all segmentations are valid, thus simply selecting the choice with the highest
        # log probability is not sufficient.


        # log probability
        logp = np.log(probs).tolist()
        
        # log complementary probability
        logcomp = np.log(1-probs).tolist()

        # cumulative log complementary probabilities used for calculating sum in token ranges
        cumul = np.cumsum(logcomp, axis=1).sum(axis=1).tolist()

        assert len(cumul) == len(logcomp), (len(cumul), len(logcomp))
        assert isinstance(cumul[0], float)

        # initialize dp array representing the highest likelyhood a segmentation can have up to the i-th token's 
        # beginning (dp[i][0]) or end (dp[i][1])
        dp = [[-float('inf')]*2 for _ in range(len(logp))]

        # used for backtracking the best segmentation
        prev = [[None]*2 for _ in range(len(logp))]

        # candidate list which contains candidate "start" indices
        # candidates = list(range(max_gap+1))  # the start index can be 0, ... , max_gap
        candidates = [0]  # the start index can be 0, ... , max_gap
        
        # logprobability threshold for considering an index as a candidate
        logp_threshold = np.log(self.candidate_threshold)

        # initialize initial values
        dp[0][0] = logp[0][0]   # --> the log-likelihood of the segmentation up to start of the 0th token

        # --> the log-likelihood of the segmentation up to the end of the 0th token
        dp[0][1] = logp[0][1] + dp[0][0]
        
        # end of the 0th token is mapped to the start of the 0th token
        prev[0][1] = 0

        for i in range(1, len(logp)):
            dp[i][0] = dp[i-1][1] + logp[i][0]
            prev[i][0] = i-1

            # update candidates list
            if logp[i][0] + logp[i-1][1] >= logp_threshold:
                candidates.append(i)
                if len(candidates) > self.max_candidates:
                    candidates.pop(0)  # remove oldest candidate

            choices = {
                k: cumul[i] - (cumul[k-1] if k else 0.) - logcomp[k][0] + dp[k][0]
                for k in candidates
            }

            # choose k with highest value
            best_k = max(choices.keys(), key=choices.get)
            best_v = max(choices.values())
            dp[i][1] = best_v + logp[i][1] - logcomp[i][1] + dp[best_k][0]
            prev[i][1] = best_k
        
        return self.backtrack(prev, binarize_output)

    def backtrack(self, prev, binarize_output):
        # backtrack to return sentence boundaries
        if binarize_output:
            bounds = [[0,0] for _ in range(len(prev))]
            i = len(prev) - 1
            while i is not None:
                bounds[i][1] = 1
                j = prev[i][1]
                bounds[j][0] = 1
                i = prev[j][0]
            return np.array(bounds)
        else:
            spans = []
            i = len(prev) - 1
            while i is not None:
                j = prev[i][1]
                spans.append((j,i+1))
                i = prev[j][0]

            return spans[::-1]


if __name__ == '__main__':
    # np.random.seed(0)
    # generate random spans
    length = 10
    x = np.random.uniform(size=(length-1,)) > 0.8
    x = x.tolist()
    row0 = [1] + x
    row1 = row0[1:] + [1]
    x = np.array([row0, row1], dtype=np.float64).T  # (N, 2)
    print(x)

    # print original boundaries
    # add some noise to binary probabilities
    noise = np.clip(np.abs(np.random.normal(0, 0.25, size=x.shape)), 0.01, 0.99)
    x[x==0] += noise[x==0]
    x[x==1] -= noise[x==1]

    # print noisy boundaries (probabilities)
    print(np.round(x, 2))

    resolver = SentenceBoundaryResolver(max_candidates=4, candidate_threshold=0.25)
    spans = resolver.resolve(x, binarize_output=True)
    
    # display resulting spans
    print(spans)

