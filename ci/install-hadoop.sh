#!/bin/bash
set -e

java -version

# remove yarnpkg to make sure hadoop yarn is called correctly
sudo apt-get remove -yq yarn || true
sudo npm uninstall -g yarn || true

sudo apt-get install -yq ssh rsync

VERSION=2.10.1
HADOOP_URL="https://www.apache.org/dyn/mirrors/mirrors.cgi?action=download&filename=hadoop/common/hadoop-$VERSION/hadoop-$VERSION.tar.gz"

# download hadoop
curl -sL "$HADOOP_URL" | tar xz --directory /tmp

# modify hadoop-env
echo "export JAVA_HOME=/usr" >> /tmp/hadoop-$VERSION/etc/hadoop/hadoop-env.sh
echo "export HADOOP_OPTS=-Djava.net.preferIPv4Stack=true" >> /tmp/hadoop-$VERSION/etc/hadoop/hadoop-env.sh

# set configuration files
cat > /tmp/hadoop-$VERSION/etc/hadoop/core-site.xml << EOF
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

cat > /tmp/hadoop-$VERSION/etc/hadoop/mapred-site.xml << EOF
<?xml version="1.0"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<configuration>
    <property>
        <name>mapred.job.tracker</name>
        <value>localhost:9010</value>
    </property>
    <property>
        <name>mapreduce.framework.name</name>
        <value>yarn</value>
    </property>
</configuration>
EOF

cat > /tmp/hadoop-$VERSION/etc/hadoop/hdfs-site.xml << EOF
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

cat > /tmp/hadoop-$VERSION/etc/hadoop/yarn-site.xml << EOF
<?xml version="1.0"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<configuration>
    <property>
        <name>yarn.resourcemanager.address</name>
        <value>localhost:8032</value>
    </property>
    <property>
        <name>yarn.nodemanager.aux-services</name>
        <value>mapreduce_shuffle</value>
    </property>
    <property>
        <name>yarn.nodemanager.aux-services.mapreduce_shuffle.class</name>
        <value>org.apache.hadoop.mapred.ShuffleHandler</value>
    </property>
    <property>
        <name>yarn.log-aggregation-enable</name>
        <value>true</value>
    </property>
    <property>
        <name>yarn.nodemanager.vmem-check-enabled</name>
        <value>false</value>
    </property>
    <property>
        <name>yarn.nodemanager.vmem-pmem-ratio</name>
        <value>4</value>
    </property>
</configuration>
EOF

cat > /tmp/hadoop.sh << EOF
#!/bin/bash
export HADOOP_HOME=/usr/local/hadoop
EOF
sudo mv /tmp/hadoop.sh /etc/profile.d/
sudo chmod a+x /etc/profile.d/hadoop.sh
sudo chown root /etc/profile.d/hadoop.sh

# create user and group
sudo addgroup hadoop
sudo adduser --disabled-password --gecos "" --ingroup hadoop hduser

sudo mkdir -p /var/lib/hadoop/tmp
sudo chmod 750 /var/lib/hadoop/tmp

sudo mkdir -p /var/lib/hadoop/hdfs/namenode
sudo mkdir -p /var/lib/hadoop/hdfs/datanode

sudo chown -R hduser:hadoop /var/lib/hadoop

# move to /usr/local
sudo mv "/tmp/hadoop-$VERSION" /usr/local
sudo ln -s "/usr/local/hadoop-$VERSION" /usr/local/hadoop
sudo chown -R hduser:hadoop "/usr/local/hadoop-$VERSION"

export HADOOP_HOME=/usr/local/hadoop

# enable ssh login without password
sudo su - hduser -c "ssh-keygen -t rsa -P \"\" -f /home/hduser/.ssh/id_rsa"
sudo su - hduser -c "cat /home/hduser/.ssh/id_rsa.pub >> /home/hduser/.ssh/authorized_keys"
sudo su - hduser -c "chmod 600 /home/hduser/.ssh/authorized_keys /home/hduser/.ssh/id_rsa"
sudo su - hduser -c "ssh -o StrictHostKeyChecking=no localhost echo "

# start hadoop
sudo su - hduser -c "$HADOOP_HOME/bin/hadoop namenode -format"
sudo su - hduser -c "$HADOOP_HOME/sbin/start-all.sh"

# create temp directory
sudo su - hduser -c "$HADOOP_HOME/bin/hdfs dfs -mkdir -p /tmp"
sudo su - hduser -c "$HADOOP_HOME/bin/hdfs dfs -chmod -R 1777 /tmp"

# create user directory
sudo su - hduser -c "$HADOOP_HOME/bin/hdfs dfs -mkdir -p /user/$USER"
sudo su - hduser -c "$HADOOP_HOME/bin/hdfs dfs -chown $USER /user/$USER"
