#!/bin/bash
set -e

sudo apt-get remove -y yarn || true
sudo apt-get update

# Installing CDH 5 with YARN on a Single Linux Node in Pseudo-distributed mode.
curl -fsSL https://archive.cloudera.com/cdh5/ubuntu/xenial/amd64/cdh/archive.key | sudo apt-key add -
echo 'deb [arch=amd64] http://archive.cloudera.com/cdh5/ubuntu/xenial/amd64/cdh xenial-cdh5 contrib' | sudo tee /etc/apt/sources.list.d/cloudera.list
echo 'deb-src http://archive.cloudera.com/cdh5/ubuntu/xenial/amd64/cdh xenial-cdh5 contrib' | sudo tee -a /etc/apt/sources.list.d/cloudera.list
sudo apt-get update
sudo apt-get -y install hadoop-conf-pseudo libhdfs0

# start a pseudo-distributed Hadoop.
sudo -u hdfs hdfs namenode -format
for x in `cd /etc/init.d ; ls hadoop-hdfs-*` ; do sudo service $x start ; done
sudo bash /usr/lib/hadoop/libexec/init-hdfs.sh

export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/jre/
