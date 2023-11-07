export ngcuser='$oauthtoken'
echo $ngcuser
oc create secret docker-registry ngc2 --docker-server=nvcr.io --docker-username='$oauthtoken' --docker-password=Y3JvdGthZnBlNmNiOGx0YWM0c285cmE4dmk6M2I5N2M0ZDktMmZhMC00ZDA3LTkyMDEtODM5MWEyMjQwMzhm --docker-email=nikolan@supermicro.com -n nvidia-gpu-operator
