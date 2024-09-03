# ACFTNet
Attentive Color Fusion Transformer Network (ACFTNet) for Underwater Image Enhancement (Code will be provided soon).

Abstract:Underwater imagery often suffers from issues like color dis-
tortion, haze, and reduced visibility due to light’s interaction with wa-
ter, posing challenges for applications like autonomous underwater ve-
hicles. To address these obstacles effectively, we introduce the Atten-
tive Color Fusion Transformer Network (ACFTNet) for underwater im-
age enhancement. At the core of our proposal lies a novel Adaptive
Dual-Gated Attentive Fusion Block (ADGAFB), which seamlessly in-
tegrates localized transmission features and global illumination charac-
teristics. Subsequently, it employs a dual-gated mechanism to generate
attentive features for each channel (R, G, and B). To ensure accurate
color fidelity, we introduce the Color-Attentive Fusion Block. This block
adeptly merges attentive features obtained from each R, G, and B chan-
nel, ensuring precise color representation. To selectively transmit features
from the encoder to the corresponding decoder, we utilize an Adaptive
Kernel-Based Channel Attention Module. Moreover, within the trans-
former block, we propose a Multi-Receptive Field Feed-Forward Gated
Network to further refine the restoration process. Through comprehen-
sive evaluations on benchmark synthetic (UIEB, EUVP) and real-world
(UIEB (challenging-60), UCCS, U45) underwater image datasets, our
method exhibits superior performance, as verified by extensive abla-
tion studies and comparative analyses.
