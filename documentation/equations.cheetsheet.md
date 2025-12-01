# LLM Inference Performance & Parallelism Cheatsheet (Unified Dense + MoE)

## 1. Key Symbols

**Model dims:**  
- $L$ (layers), $H$ (hidden size), $n_q$ (query heads), $n_{kv}$ (KV heads),  
- $d_{\text{head}} = H/n_q$, $H_{kv} = n_{kv} d_{\text{head}}$  
- $I$ (FFN dim), $I_{\text{eff}} = I_{\text{dense}}$ or $k I_{\text{moe}}$  
- $N_{\text{eff}} = 0$ (dense) or $N_{\text{exp}}$ (MoE)

**Parallelism:**  
$DP \rightarrow PP \rightarrow EP \rightarrow TP \rightarrow SP$

**Sequence / precision:**  
$S$ (context), $b$ (bytes/elt), $c_{\text{act}}\approx 8$–$12$

---

## 2. Per-Layer Parameters, Activations, KV

**Attention params**  
$$P_{\text{attn}} = H^2 + 3 H H_{kv}$$

**FFN params (unified)**  
$$P_{\text{FFN}} = 3 H I N_{\text{exp}}$$

**Activations**  
$$P_{\text{act}} = 4H + 2H_{kv}$$

**KV cache**  
$$P_{\text{KV,layer}} = 2 S H_{kv}$$

---

## 3. Per-Device Static Memory

**Parameter memory**  
$$
M_{\theta,\text{device}} =
\frac{L}{PP}
\left(
\frac{H^2 + 3HH_{kv}}{TP}
+ \frac{3H I N_{\text{exp}}}{TP\cdot EP}
\right)b
+ \frac{VH}{TP}b
$$

**Activation memory**  
$$
M_{\text{act,device}} = 
\frac{L}{PP}(4H + 2H_{kv}) b
$$

**KV memory**  
$$
M_{\text{KV,device}} =
\frac{L}{PP} \cdot 
\frac{2 S H_{kv} b}{TP \cdot SP}
$$

**Total**  
$$
M_{\text{device}}^{\text{total}} =
M_{\theta,\text{device}}
+ M_{\text{act,device}}
+ M_{\text{KV,device}}
$$

---

## 4. Per-Token Memory Traffic

**Parameter traffic**  
$$
T_{\theta} \approx
\frac{L}{PP}
\left(
\frac{H^2 + 3HH_{kv}}{TP\gamma_{FA}}
+ \frac{3H I N_{\text{exp}}}{TP\cdot EP\cdot \gamma_{FMLP}}
\right) b
$$

**Activation traffic**  
$$
T_{\text{act}} =
\frac{L}{PP} c_{\text{act}} H b
$$

**KV traffic**  
$$
T_{\text{KV}} =
\frac{L}{PP}
\frac{2 S H_{kv} b}{TP \cdot SP}
$$

**Total**  
$$
T_{\text{token,device}}^{eff}
\approx
\frac{L}{PP}
\left(
\frac{H^2 + 3HH_{kv}}{TP\cdot \gamma_{FA}}
+ \frac{3HI N_{\text{exp}}}{TP\cdot EP\cdot \gamma_{FMLP}}
+ c_{\text{act}}H
+ \frac{2 S H_{kv}}{TP\cdot SP}
\right)b
$$

---

## 5. Per-Token FLOPs

**Projections**  
$$F_{\text{proj}} = 2H^2 + 6HH_{kv}$$

**Attention score + value**  
$$F_{\text{attn,KV}} = 4 S H_{kv}$$

**Unified FFN FLOPs**  
$$F_{\text{ffn}} = 4H I_{\text{eff}} + 2H N_{\text{eff}}$$

**Per-device FLOPs**  
$$
F_{\text{token,device}} \approx
\frac{L}{PP}
\left(
\frac{2H^2 + 6HH_{kv}}{TP}
+ \frac{4H I_{\text{eff}}}{TP\cdot EP}
+ \frac{4 S H_{kv}}{TP\cdot SP}
+ 2H N_{\text{eff}}
\right)
$$

---

## 6. Local Latency (Roofline)

**Compute time**  
$$t_{\text{compute}} = \frac{F_{\text{token,device}}}{R_{\text{GPU}}}$$

**Memory time**  
$$t_{\text{mem}} = \frac{T_{\text{token,device}}^{eff}}{B_{\text{eff,mem}}}$$

**Local bottleneck**  
$$
t_{\text{local}} = \max(t_{\text{compute}},\, t_{\text{mem}})
$$

**Operational Intensity**  
$$
\text{OI} = \frac{F_{\text{token,device}}}{T_{\text{token,device}}^{eff}}
\approx \frac{2}{b} \quad\text{(long context)}
$$

---

## 7. Communication Costs (Per Token)

### PP hop
$$
t_{PP} = \alpha_{PP} + \frac{(H/TP)b}{B_{\text{eff,PP}}}
$$

### EP all-to-all (1 pass)
Ring:
$$
t_{EP}^{\text{ring}} =
(EP-1)\alpha_{EP}
+ (EP-1)\frac{kH b}{EP B_{\text{eff,EP}}}
$$

Tree:
$$
t_{EP}^{\text{tree}} \approx
\lceil\log_2 EP\rceil\alpha_{EP}
+ \frac{kH b}{B_{\text{eff,EP}}}
$$

### TP all-reduce (2 pass)
Ring:
$$
t_{TP}^{\text{ring}} =
2(TP-1)\alpha_{TP}
+2\frac{TP-1}{TP}\frac{(H/TP)b}{B_{\text{eff,TP}}}
$$

Tree:
$$
t_{TP}^{\text{tree}} =
2\lceil\log_2 TP\rceil\alpha_{TP}
+2\frac{(H/TP)b}{B_{\text{eff,TP}}}
$$

### SP ring (2 pass)
$$
t_{SP} =
2(SP-1)\alpha_{SP}
+ 2\frac{SP-1}{SP}
\frac{\left(\frac{S}{SP}\cdot\frac{2H_{kv}}{TP}\right)b}
{B_{\text{eff,SP}}}
$$

### Total PP-stage communication
$$
t_{\text{comm}} =
\frac{L}{PP}(n_{TP}t_{TP} + n_{EP}t_{EP} + t_{SP})
+ t_{PP}
$$

---

## 8. Overlap Model

$$
t_{\text{token}}^{\text{no}} = t_{\text{local}} + t_{\text{comm}}
$$

$$
t_{\text{token}}^{\text{full}} = \max(t_{\text{local}},\,t_{\text{comm}})
$$

$$
t_{\text{token}} \approx \max(t_{\text{local}},\, \rho t_{\text{comm}}),
\qquad \rho\in[0.3,1]
$$

---

## 9. TPS / TTPS

**Stage latency**  
$$t_{\text{stage},j} = \max(t_{\text{local},j},\,\rho t_{\text{comm},j})$$

**Single-replica TPS**  
$$
TPS_{\text{single}} = 
\frac{1}{\max_j t_{\text{stage},j}}
$$

**Total cluster throughput**  
$$
TTPS = DP \cdot TPS_{\text{single}}
$$

---

## 10. TTFT

$$
TTFT \approx
t_{\text{prefill}}
+ t_{\text{KV-transfer}}
+ \sum_{j=1}^{PP} t_{\text{stage},j}
+ t_{\text{startup}}
$$

**KV transfer**  
$$
t_{\text{KV-transfer}} \approx
\frac{2 S H_{kv} b L}{B_{\text{link,cluster}}}
$$
