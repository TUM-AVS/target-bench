### Average Displacement Error (ADE)

The **Average Displacement Error (ADE)** measures the average L2 distance between predicted and ground-truth trajectories over all time steps.  
It captures the overall accuracy of a predicted trajectory in matching the ground truth path.

We define it as:

\[
\text{ADE} = \frac{1}{T} \sum_{t=1}^{T} \| \hat{s}_t - s_t \|_2
\]

where \( T \) is the total number of time steps,  
\( \hat{s}_t \) is the predicted position at time \( t \),  
and \( s_t \) is the corresponding ground truth position.


### Final Displacement Error (FDE)

The **Final Displacement Error (FDE)** measures the L2 distance between the predicted final position and the ground-truth final position.  
It focuses on the accuracy of the final predicted point, which is particularly important for long-horizon trajectory prediction.

We define it as:

\[
\text{FDE} = \| \hat{s}_T - s_T \|_2
\]

where \( \hat{s}_T \) and \( s_T \) denote the predicted and ground-truth positions at the final time step \( T \).


### Miss rate (MR)

A binary match/miss indicator function \( \text{IsMATCH}(\hat{s}_t, s_t) \) is assigned to each sample waypoint at a time \( t \).  
The average over the dataset creates the **miss rate** at that time step.  

A single distance threshold to determine IsMATCH is insufficient: we want a stricter criteria for slower moving and closer-in-time predictions, and also different criteria for lateral deviation (e.g., wrong lane) versus longitudinal (e.g., wrong speed profile).  

We define it as:

\[
IsMatch(\hat{s}_t, s_t) = \mathbf{1}[x_t^k < \lambda^{lon}] \cdot \mathbf{1}[y_t^k < \lambda^{lat}]
\tag{1}
\]

\[
[x_t^k, y_t^k] := (\hat{s}_t - s_t^k) \cdot R_t
\]

where \( R_t \) is a 2D rotation matrix defined by the ground truth heading of the agent at timestamp \( t \).  
The parameters \( \lambda^{lon} \) and \( \lambda^{lat} \) are longitudinal and lateral thresholds.


### Soft Endpoint (SE)

The **Soft Endpoint (SE)** metric measures how close the predicted trajectory’s final point is to the ground-truth endpoint, using a Gaussian-shaped distance penalty.  
It rewards predictions that end *near* the goal rather than requiring an exact match, thus providing a smoother evaluation than FDE.

We define it as:

\[
\text{SE} = \exp\!\Big(-\frac{\|\hat{s}_T - s_T^{GT}\|_2^2}{2\sigma^2}\Big)
\]

where \( \hat{s}_T \) and \( s_T^{GT} \) are the predicted and ground-truth endpoints, respectively, and \( \sigma \) controls the tolerance of acceptable endpoint deviation.  
In our experiments, we set \( \sigma = 0.6 \) meters.

SE returns a score in \([0,1]\), where 1 indicates perfect endpoint alignment and smaller values correspond to greater endpoint error.  
This metric focuses purely on **goal-reaching accuracy** and is invariant to the specific path taken.


### Approach Consistency (AC)

The **Approach Consistency (AC)** metric evaluates whether the predicted trajectory maintains a generally similar *approaching tendency* toward the target as the ground-truth path, while allowing flexibility in the exact route.

#### Corridor construction
1. Uniformly sample **20 reference points** along the ground-truth trajectory:
   \[
   s^{GT}_1, s^{GT}_2, \dots, s^{GT}_{20}
   \]
2. Assign each point a variable radius to form a **progress-dependent corridor**:
   \[
   \sigma_i = \sigma_{\min} + (\sigma_{\max} - \sigma_{\min}) 
              \cdot \exp\!\Big(-\frac{(p_i - 0.5)^2}{2\beta^2}\Big)
   \]
   where \( p_i \in [0,1] \) represents normalized progress along the trajectory.  
   Typically, \( \sigma_{\min}=0.15\,\text{m} \), \( \sigma_{\max}=0.5\,\text{m} \), and \( \beta=0.25 \).  
   This yields a **narrow corridor** at the start and goal, and a **wider corridor** in the middle.

#### Coverage evaluation
For each predicted point \( \hat{s}_j \), compute its distance to all corridor centers:
\[
d_{ij} = \|\hat{s}_j - s^{GT}_i\|_2
\]
If \( d_{ij} \le \sigma_i \) for any \( i \), then \( \hat{s}_j \) is considered **covered**.

Let \( N_{\text{pred}} \) denote the number of predicted points (e.g., 9) and \( N_{\text{covered}} \) the number of predicted points falling within at least one corridor region.

#### AC computation
\[
\text{AC} = 
\begin{cases}
1, & N_{\text{covered}} = N_{\text{pred}} \\
\exp\!\Big(-\gamma \cdot \frac{N_{\text{pred}} - N_{\text{covered}}}{N_{\text{pred}}}\Big), & \text{otherwise}
\end{cases}
\]

where \( \gamma \) controls the penalty sharpness (e.g., \( \gamma = 5 \)).

A high AC indicates that the predicted trajectory remains within the general corridor of the ground-truth path (reflecting similar approach behavior), while lower values signify significant deviation from the intended approach.


### Overall Trajectory Evaluation Score (Overall)

This document defines an **Overall Score** that aggregates multiple trajectory evaluation metrics into a single, interpretable value within the range **[0, 1]** (higher is better).

---

#### 1. Objective

To fairly evaluate predicted trajectories by combining several complementary metrics — accuracy, goal-reaching, smoothness, and behavioral consistency — into one unified score.

\[
S = \sum_i w_i \cdot s_i, \qquad \sum_i w_i = 1,\; w_i \ge 0
\]

where  
- \( s_i \) = normalized sub-score of each metric (mapped to [0,1], higher = better)  
- \( w_i \) = corresponding weight

---

#### 2. Component Metrics

| Metric | Description | Direction | Normalization |
|:-------:|--------------|------------|----------------|
| **ADE** | Average Displacement Error | Lower is better | \( s_{\text{ADE}} = \exp(-\frac{\text{ADE}}{\tau_{\text{ADE}}}) \) |
| **FDE** | Final Displacement Error | Lower is better | \( s_{\text{FDE}} = \exp(-\frac{\text{FDE}}{\tau_{\text{FDE}}}) \) |
| **MR**  | Miss Rate | Lower is better | \( s_{\text{MR}} = 1 - \text{MR} \) |
| **SE**  | Soft Endpoint | Higher is better | \( s_{\text{SE}} = \text{SE} \) |
| **AC**  | Approach Consistency | Higher is better | \( s_{\text{AC}} = \text{AC} \) |

Recommended scale parameters:
\[
\tau_{\text{ADE}} = 1.0\ \text{m}, \quad \tau_{\text{FDE}} = 1.0\ \text{m}
\]

---

#### 3. Default Weight Configuration

Balanced configuration (accuracy + goal-reaching + consistency):

| Metric | Weight |
|:--------|:-------:|
| ADE | 0.05 |
| FDE | 0.10 |
| MR  | 0.10 |
| SE  | 0.35 |
| AC  | 0.30 |

Weights sum to 1.

---

#### 4. Score Formula

\[
\begin{aligned}
S_{\text{overall}} &=
w_{\text{ADE}} \cdot e^{-\frac{\text{ADE}}{\tau_{\text{ADE}}}} +
w_{\text{FDE}} \cdot e^{-\frac{\text{FDE}}{\tau_{\text{FDE}}}} +
w_{\text{MR}}  \cdot (1 - \text{MR}) \\
&\quad + (w_{\text{SE}} + w_{\text{AC}}) \cdot \text{SE} \cdot \text{AC}
\end{aligned}
\]

\[
S_{\text{overall}} \in [0,1]
\]
