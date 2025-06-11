import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/wwojtak/time4hri_dnf/install/dnf_cognitive_architecture_extended'
