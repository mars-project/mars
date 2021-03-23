#!/bin/bash
set -e
export CHANGE_MINIKUBE_NONE_USER=true

sudo apt-get -q update || true
sudo apt-get install -yq conntrack jq

get_latest_release() {
  curl --silent "https://api.github.com/repos/$1/releases" |
    jq -c '[.[] | select(.prerelease == false)][0].tag_name' |
    sed -E 's/.*"([^"]+)".*/\1/'
}

K8S_VERSION=$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)
if [[ "$K8S_VERSION" == *"alpha"* ]] || [[ "$K8S_VERSION" == *"beta"* ]] || [[ "$K8S_VERSION" == *"rc"* ]]; then
  K8S_VERSION=$(get_latest_release "kubernetes/kubernetes")
fi

curl -Lo kubectl https://storage.googleapis.com/kubernetes-release/release/$K8S_VERSION/bin/linux/amd64/kubectl && \
  chmod +x kubectl && sudo mv kubectl /usr/local/bin/

curl -Lo minikube https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64 && \
  chmod +x minikube && sudo mv minikube /usr/local/bin/

sudo minikube start --vm-driver=none --kubernetes-version=$K8S_VERSION
export KUBECONFIG=$HOME/.kube/config
sudo cp -R /root/.kube /root/.minikube $HOME/
sudo chown -R $(id -u):$(id -g) $HOME/.kube $HOME/.minikube

sed "s/root/home\/$USER/g" $KUBECONFIG > tmp
mv tmp $KUBECONFIG

minikube update-context

JSONPATH='{range .items[*]}{@.metadata.name}:{range @.status.conditions[*]}{@.type}={@.status};{end}{end}'
until kubectl get nodes -o jsonpath="$JSONPATH" 2>&1 | grep -q "Ready=True"; do
  sleep 1
done
