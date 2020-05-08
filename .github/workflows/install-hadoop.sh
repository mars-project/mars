#!/bin/bash
set -e

java -version

sudo apt-get remove -yq yarn || true
sudo apt-get install -yq ssh rsync

VERSION=hadoop-2.10.0
HADOOP_URL="https://www.apache.org/dyn/mirrors/mirrors.cgi?action=download&filename=hadoop/common/$VERSION/$VERSION.tar.gz"

# download hadoop
curl -sL "$HADOOP_URL" | tar xz --directory /tmp

# modify hadoop-env
echo "export JAVA_HOME=/usr" >> /tmp/$VERSION/etc/hadoop/hadoop-env.sh
echo "export HADOOP_OPTS=-Djava.net.preferIPv4Stack=true" >> /tmp/$VERSION/etc/hadoop/hadoop-env.sh

# set configuration files
cat > /tmp/$VERSION/etc/hadoop/core-site.xml << EOF
<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<configuration>
    <property>
        <name>hadoop.tmp.dir</name>
        <value>/var/lib/hadoop/tmp</value>
        <description>A base for other temporary directories.</description>
    </property>
    <property>
        <name>fs.default.name</name>
        <value>hdfs://localhost:8020</value>
    </property>
</configuration>
EOF

cat > /tmp/$VERSION/etc/hadoop/mapred-site.xml << EOF
<?xml version="1.0"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<configuration>
    <property>
        <name>mapred.job.tracker</name>
        <value>localhost:9010</value>
    </property>
</configuration>
EOF

cat > /tmp/$VERSION/etc/hadoop/hdfs-site.xml << EOF
<?xml version="1.0"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<configuration>
    <property>
        <name>dfs.secondary.http.address</name>
        <value>localhost:50090</value>
    </property>
    <property>
        <name>dfs.replication</name>
        <value>1</value>
    </property>
    <property>
        <name>dfs.namenode.name.dir</name>
        <value>file:/var/lib/hadoop/hdfs/namenode</value>
    </property>
    <property>
        <name>dfs.datanode.data.dir</name>
        <value>file:/var/lib/hadoop/hdfs/datanode</value>
    </property>
</configuration>
EOF

# create user and group
sudo addgroup hadoop
sudo adduser --disabled-password --gecos "" --ingroup hadoop hduser

sudo mkdir -p /var/lib/hadoop/tmp
sudo chmod 750 /var/lib/hadoop/tmp

sudo mkdir -p /var/lib/hadoop/hdfs/namenode
sudo mkdir -p /var/lib/hadoop/hdfs/datanode

sudo chown -R hduser:hadoop /var/lib/hadoop

# move to /usr/local
sudo mv "/tmp/$VERSION" /usr/local
sudo ln -s "/usr/local/$VERSION" /usr/local/hadoop
sudo chown -R hduser:hadoop "/usr/local/$VERSION"

# enable ssh login without password
sudo su - hduser -c "ssh-keygen -t rsa -P \"\" -f /home/hduser/.ssh/id_rsa"
sudo su - hduser -c "cat /home/hduser/.ssh/id_rsa.pub >> /home/hduser/.ssh/authorized_keys"
sudo su - hduser -c "chmod 600 /home/hduser/.ssh/authorized_keys"
sudo su - hduser -c "ssh -o StrictHostKeyChecking=no localhost echo "

# start hadoop
sudo su - hduser -c "/usr/local/hadoop/bin/hadoop namenode -format"
sudo su - hduser -c "/usr/local/hadoop/sbin/start-all.sh"

# create temp directory
sudo su - hduser -c "/usr/local/hadoop/bin/hdfs dfs -mkdir -p /tmp"
sudo su - hduser -c "/usr/local/hadoop/bin/hdfs dfs -chmod -R 1777 /tmp"
sudo su - hduser -c "/usr/local/hadoop/bin/hdfs dfs -ls /"
