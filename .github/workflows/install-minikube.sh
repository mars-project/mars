#!/bin/bash
set -e
export CHANGE_MINIKUBE_NONE_USER=true

sudo apt-get -q update || true
sudo apt-get install -yq conntrack jq

curl -Lo minikube https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64 && \
  chmod +x minikube && sudo mv minikube /usr/local/bin/

sudo minikube start --vm-driver=none
export KUBECONFIG=$HOME/.kube/config
sudo cp -R /root/.kube /root/.minikube $HOME/
sudo chown -R $(id -u):$(id -g) $HOME/.kube $HOME/.minikube

sed "s/root/home\/$USER/g" $KUBECONFIG > tmp
mv tmp $KUBECONFIG

minikube update-context

K8S_VERSION=$(minikube kubectl -- version --client --output='json' | jq -r '.clientVersion.gitVersion')
curl -Lo kubectl https://storage.googleapis.com/kubernetes-release/release/$K8S_VERSION/bin/linux/amd64/kubectl && \
  chmod +x kubectl && sudo mv kubectl /usr/local/bin/

JSONPATH='{range .items[*]}{@.metadata.name}:{range @.status.conditions[*]}{@.type}={@.status};{end}{end}'
until kubectl get nodes -o jsonpath="$JSONPATH" 2>&1 | grep -q "Ready=True"; do
  sleep 1
done
