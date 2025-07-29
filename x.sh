set -eux

# 编译
python3 setup.py develop

# 使用cpu
python3 test.py

# 使用virtd，了解分派机制
python3 test.py

