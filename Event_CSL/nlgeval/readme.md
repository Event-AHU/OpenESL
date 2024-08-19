pip install git+https://github.com/Maluuba/nlg-eval.git@master



If you have this problem:
No such file or directory: 'java'

Download java package: 
link：https://pan.baidu.com/s/1ZbfVS-xl6Zd8f0akXeKJ5g?pwd=EAHU 
pwd：EAHU

```bash
mv jdk-17.0.4 /home/username/
```
Grant read and write permissions if necessary:
```bash
sudo chmod -R 777 jdk-17.0.4
```
After that, you need to add the following lines to your ```~/.bashrc```:
```bash
export JAVA_HOME=/homedata/username/jdk-17.0.4
export PATH=$PATH:$JAVA_HOME/bin
export CLASSPATH=.:$JAVA_HOME/lib/dt.jar:$JAVA_HOME/lib/tools.jar
export JRE_HOME=$JAVA_HOME/jre
```
Finally execute the following command:
```bash
source ~/.bashrc
```
