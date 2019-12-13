#!/bin/bash
set -e
export CHANGE_MINIKUBE_NONE_USER=true

sudo apt-get remove -y docker.io || true
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get -q update || true
sudo apt-get install -yq docker-ce

K8S_VERSION=$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)

curl -Lo kubectl https://storage.googleapis.com/kubernetes-release/release/$K8S_VERSION/bin/linux/amd64/kubectl && \
  chmod +x kubectl && sudo mv kubectl /usr/local/bin/

curl -Lo minikube https://storage.googleapis.com/minikube/releases/v1.3.1/minikube-linux-amd64 && \
  chmod +x minikube && sudo mv minikube /usr/local/bin/

sudo minikube start --vm-driver=none --kubernetes-version=$K8S_VERSION
sudo chown -R $(id -u):$(id -g) $HOME/.minikube
sudo chown -R $(id -u):$(id -g) $HOME/.kube

minikube update-context

JSONPATH='{range .items[*]}{@.metadata.name}:{range @.status.conditions[*]}{@.type}={@.status};{end}{end}'
until kubectl get nodes -o jsonpath="$JSONPATH" 2>&1 | grep -q "Ready=True"; do
  sleep 1
done
