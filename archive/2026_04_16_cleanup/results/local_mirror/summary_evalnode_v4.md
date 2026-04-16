# Control-V5 Eval Summary

| model                               |   acc |   conf_cov |     ece |   brier |   wrong_hi |   meta |   verify |   redirect |   diagnosis |   study_need |
|:------------------------------------|------:|-----------:|--------:|--------:|-----------:|-------:|---------:|-----------:|------------:|-------------:|
| control_v5_E3                       | 0.300 |      1.000 |   0.515 |   0.455 |      0.714 |  6.444 |    0.256 |      0.233 |       0.089 |        0.156 |
| control_v5_E5                       | 0.400 |      0.011 |   0.790 |   0.624 |      0.000 |  0.011 |    0.000 |      0.011 |       0.056 |        0.011 |
| control_v5_E9                       | 0.411 |      0.911 |   0.300 |   0.305 |      0.509 |  0.956 |    0.678 |      0.233 |       0.233 |        0.156 |
| qwen3_base_sft                      | 0.422 |      0.000 | nan     | nan     |      0.000 |  0.000 |    0.000 |      0.000 |       0.044 |        0.000 |
| qwen3_metacot_control_v5_all_sft    | 0.333 |      0.889 |   0.398 |   0.347 |      0.517 |  0.889 |    0.167 |      0.256 |       0.189 |        0.300 |
| qwen3_metacot_control_v5_verify_sft | 0.367 |      0.822 |   0.492 |   0.466 |      0.807 |  0.833 |    0.133 |      0.000 |       0.044 |        0.000 |

## control_v5_E3
- `aime2024`: acc=0.033, conf_cov=1.000, ece=0.726, brier=0.581, wrong_hi=0.621, meta=7.667, verify=0.233, redirect=0.400, diagnosis=0.167
- `gsm8k`: acc=0.633, conf_cov=1.000, ece=0.243, brier=0.289, wrong_hi=1.000, meta=5.100, verify=0.167, redirect=0.000, diagnosis=0.000
- `math500`: acc=0.233, conf_cov=1.000, ece=0.576, brier=0.495, wrong_hi=0.696, meta=6.567, verify=0.367, redirect=0.300, diagnosis=0.100

### Overconfident Wrong Samples
- `gsm8k` conf=0.875 | Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased 
- `gsm8k` conf=0.870 | Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables
- `gsm8k` conf=0.875 | Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulo
- `gsm8k` conf=0.871 | Carla is downloading a 200 GB file. Normally she can download 2 GB/minute, but 40% of the way through the download, Wind
- `gsm8k` conf=0.879 | John drives for 3 hours at a speed of 60 mph and then turns around because he realizes he forgot something very importan

### AIME Behavior Samples
- correct=False conf=0.871 verify=False redirect=False diagnosis=False | Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant
- correct=False conf=0.875 verify=True redirect=False diagnosis=False | Let $ABC$ be a triangle inscribed in circle $\omega$. Let the tangents to $\omega$ at $B$ and $C$ intersect at point $D$
- correct=False conf=0.266 verify=False redirect=True diagnosis=True | Each vertex of a regular octagon is independently colored either red or blue with equal probability. The probability tha
- correct=False conf=0.793 verify=False redirect=True diagnosis=False | Define $f(x)=|| x|-\tfrac{1}{2}|$ and $g(x)=|| x|-\tfrac{1}{4}|$. Find the number of intersections of the graphs of \[y=
- correct=False conf=0.705 verify=False redirect=True diagnosis=False | Let $p$ be the least prime number for which there exists a positive integer $n$ such that $n^{4}+1$ is divisible by $p^{

## control_v5_E5
- `aime2024`: acc=0.033, conf_cov=0.033, ece=0.790, brier=0.624, wrong_hi=0.000, meta=0.033, verify=0.000, redirect=0.033, diagnosis=0.133
- `gsm8k`: acc=0.800, conf_cov=0.000, ece=n/a, brier=n/a, wrong_hi=0.000, meta=0.000, verify=0.000, redirect=0.000, diagnosis=0.000
- `math500`: acc=0.367, conf_cov=0.000, ece=n/a, brier=n/a, wrong_hi=0.000, meta=0.000, verify=0.000, redirect=0.000, diagnosis=0.033

### AIME Behavior Samples
- correct=False conf=n/a verify=False redirect=False diagnosis=False | Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant
- correct=False conf=n/a verify=False redirect=False diagnosis=True | Let $ABC$ be a triangle inscribed in circle $\omega$. Let the tangents to $\omega$ at $B$ and $C$ intersect at point $D$
- correct=False conf=n/a verify=False redirect=False diagnosis=False | Each vertex of a regular octagon is independently colored either red or blue with equal probability. The probability tha
- correct=False conf=n/a verify=False redirect=False diagnosis=False | Define $f(x)=|| x|-\tfrac{1}{2}|$ and $g(x)=|| x|-\tfrac{1}{4}|$. Find the number of intersections of the graphs of \[y=
- correct=False conf=n/a verify=False redirect=False diagnosis=False | Let $p$ be the least prime number for which there exists a positive integer $n$ such that $n^{4}+1$ is divisible by $p^{

## control_v5_E9
- `aime2024`: acc=0.067, conf_cov=0.800, ece=0.411, brier=0.330, wrong_hi=0.250, meta=0.900, verify=0.233, redirect=0.533, diagnosis=0.567
- `gsm8k`: acc=0.900, conf_cov=1.000, ece=0.017, brier=0.088, wrong_hi=1.000, meta=1.000, verify=1.000, redirect=0.000, diagnosis=0.000
- `math500`: acc=0.267, conf_cov=0.933, ece=0.546, brier=0.516, wrong_hi=0.773, meta=0.967, verify=0.800, redirect=0.167, diagnosis=0.133

### Overconfident Wrong Samples
- `gsm8k` conf=0.860 | Carla is downloading a 200 GB file. Normally she can download 2 GB/minute, but 40% of the way through the download, Wind
- `gsm8k` conf=0.880 | In a dance class of 20 students, 20% enrolled in contemporary dance, 25% of the remaining enrolled in jazz dance, and th
- `gsm8k` conf=0.880 | Raymond and Samantha are cousins. Raymond was born 6 years before Samantha. Raymond had a son at the age of 23. If Saman
- `math500` conf=0.900 | Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\theta),$ 
- `math500` conf=0.880 | Define
\[p = \sum_{k = 1}^\infty \frac{1}{k^2} \quad \text{and} \quad q = \sum_{k = 1}^\infty \frac{1}{k^3}.\]Find a way

### AIME Behavior Samples
- correct=False conf=0.320 verify=False redirect=True diagnosis=True | Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant
- correct=False conf=0.340 verify=False redirect=True diagnosis=True | Let $ABC$ be a triangle inscribed in circle $\omega$. Let the tangents to $\omega$ at $B$ and $C$ intersect at point $D$
- correct=False conf=0.340 verify=False redirect=True diagnosis=True | Each vertex of a regular octagon is independently colored either red or blue with equal probability. The probability tha
- correct=False conf=0.340 verify=False redirect=True diagnosis=True | Define $f(x)=|| x|-\tfrac{1}{2}|$ and $g(x)=|| x|-\tfrac{1}{4}|$. Find the number of intersections of the graphs of \[y=
- correct=False conf=0.880 verify=True redirect=False diagnosis=False | Let $p$ be the least prime number for which there exists a positive integer $n$ such that $n^{4}+1$ is divisible by $p^{

## qwen3_base_sft
- `aime2024`: acc=0.133, conf_cov=0.000, ece=n/a, brier=n/a, wrong_hi=0.000, meta=0.000, verify=0.000, redirect=0.000, diagnosis=0.100
- `gsm8k`: acc=0.800, conf_cov=0.000, ece=n/a, brier=n/a, wrong_hi=0.000, meta=0.000, verify=0.000, redirect=0.000, diagnosis=0.000
- `math500`: acc=0.333, conf_cov=0.000, ece=n/a, brier=n/a, wrong_hi=0.000, meta=0.000, verify=0.000, redirect=0.000, diagnosis=0.033

### AIME Behavior Samples
- correct=False conf=n/a verify=False redirect=False diagnosis=False | Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant
- correct=False conf=n/a verify=False redirect=False diagnosis=False | Let $ABC$ be a triangle inscribed in circle $\omega$. Let the tangents to $\omega$ at $B$ and $C$ intersect at point $D$
- correct=False conf=n/a verify=False redirect=False diagnosis=False | Each vertex of a regular octagon is independently colored either red or blue with equal probability. The probability tha
- correct=False conf=n/a verify=False redirect=False diagnosis=False | Define $f(x)=|| x|-\tfrac{1}{2}|$ and $g(x)=|| x|-\tfrac{1}{4}|$. Find the number of intersections of the graphs of \[y=
- correct=False conf=n/a verify=False redirect=False diagnosis=False | Let $p$ be the least prime number for which there exists a positive integer $n$ such that $n^{4}+1$ is divisible by $p^{

## qwen3_metacot_control_v5_all_sft
- `aime2024`: acc=0.067, conf_cov=1.000, ece=0.478, brier=0.318, wrong_hi=0.357, meta=1.000, verify=0.267, redirect=0.500, diagnosis=0.367
- `gsm8k`: acc=0.633, conf_cov=0.733, ece=0.324, brier=0.343, wrong_hi=0.818, meta=0.733, verify=0.133, redirect=0.033, diagnosis=0.033
- `math500`: acc=0.300, conf_cov=0.933, ece=0.417, brier=0.383, wrong_hi=0.571, meta=0.933, verify=0.100, redirect=0.233, diagnosis=0.167

### Overconfident Wrong Samples
- `gsm8k` conf=0.880 | Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables
- `gsm8k` conf=0.860 | Carla is downloading a 200 GB file. Normally she can download 2 GB/minute, but 40% of the way through the download, Wind
- `gsm8k` conf=0.880 | John drives for 3 hours at a speed of 60 mph and then turns around because he realizes he forgot something very importan
- `gsm8k` conf=0.880 | Eliza's rate per hour for the first 40 hours she works each week is $10. She also receives an overtime pay of 1.2 times 
- `gsm8k` conf=0.880 | Toula went to the bakery and bought various types of pastries. She bought 3 dozen donuts which cost $68 per dozen, 2 doz

### AIME Behavior Samples
- correct=False conf=0.340 verify=True redirect=False diagnosis=True | Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant
- correct=False conf=0.340 verify=True redirect=True diagnosis=True | Let $ABC$ be a triangle inscribed in circle $\omega$. Let the tangents to $\omega$ at $B$ and $C$ intersect at point $D$
- correct=False conf=0.860 verify=False redirect=False diagnosis=False | Each vertex of a regular octagon is independently colored either red or blue with equal probability. The probability tha
- correct=False conf=0.310 verify=False redirect=True diagnosis=True | Define $f(x)=|| x|-\tfrac{1}{2}|$ and $g(x)=|| x|-\tfrac{1}{4}|$. Find the number of intersections of the graphs of \[y=
- correct=False conf=0.340 verify=False redirect=True diagnosis=False | Let $p$ be the least prime number for which there exists a positive integer $n$ such that $n^{4}+1$ is divisible by $p^{

## qwen3_metacot_control_v5_verify_sft
- `aime2024`: acc=0.033, conf_cov=0.833, ece=0.813, brier=0.700, wrong_hi=0.828, meta=0.867, verify=0.267, redirect=0.000, diagnosis=0.100
- `gsm8k`: acc=0.667, conf_cov=0.767, ece=0.148, brier=0.208, wrong_hi=0.600, meta=0.767, verify=0.000, redirect=0.000, diagnosis=0.000
- `math500`: acc=0.400, conf_cov=0.867, ece=0.488, brier=0.470, wrong_hi=0.889, meta=0.867, verify=0.133, redirect=0.000, diagnosis=0.033

### Overconfident Wrong Samples
- `gsm8k` conf=0.860 | Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables
- `gsm8k` conf=0.880 | Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulo
- `gsm8k` conf=0.860 | Carla is downloading a 200 GB file. Normally she can download 2 GB/minute, but 40% of the way through the download, Wind
- `gsm8k` conf=0.880 | Carlos is planting a lemon tree. The tree will cost $90 to plant. Each year it will grow 7 lemons, which he can sell for
- `gsm8k` conf=0.860 | Melanie is a door-to-door saleswoman. She sold a third of her vacuum cleaners at the green house, 2 more to the red hous

### AIME Behavior Samples
- correct=False conf=0.840 verify=True redirect=False diagnosis=False | Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant
- correct=False conf=0.840 verify=False redirect=False diagnosis=False | Let $ABC$ be a triangle inscribed in circle $\omega$. Let the tangents to $\omega$ at $B$ and $C$ intersect at point $D$
- correct=False conf=0.860 verify=False redirect=False diagnosis=False | Each vertex of a regular octagon is independently colored either red or blue with equal probability. The probability tha
- correct=False conf=0.840 verify=True redirect=False diagnosis=False | Define $f(x)=|| x|-\tfrac{1}{2}|$ and $g(x)=|| x|-\tfrac{1}{4}|$. Find the number of intersections of the graphs of \[y=
- correct=False conf=n/a verify=False redirect=False diagnosis=False | Let $p$ be the least prime number for which there exists a positive integer $n$ such that $n^{4}+1$ is divisible by $p^{
