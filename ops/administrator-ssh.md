# Administrator action: unattended LOTAAS SSH

The head node needs to reconnect to the compute nodes after a dropped SSH
master or reboot, without a forwarded laptop agent. The compute sshd reads
`/etc/ssh/authorized_keys/%u` and SSSD, not user `~/.ssh/authorized_keys`.

Please append this public-key entry to `/etc/ssh/authorized_keys/dkuiper`
(or apply an equivalent central-account entry with the same restrictions)
on efc-gpu-00, efc-gpu-01, and efc-cpu-00 through efc-cpu-06:

```
restrict,from="2001:610:568:2280::4" ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIIKHTepHhpxw4fHg8s0D558ZPrYZuzGvmN+FenW2ti9E lotaas-head-unattended
```

Preserve existing authorized keys. No private key needs to leave efc-head.
The new key is `/home/dkuiper/.ssh/id_ed25519_lotaas_cluster` there, mode 0600.

Verification from efc-head:

```
ssh -o BatchMode=yes -o ControlPath=none -o IdentityAgent=none \
  -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519_lotaas_cluster efc-gpu-00 true
```

Repeat for the remaining nodes. The current SSH masters are working, so this
can be done while the campaign runs. No SSH daemon configuration change or
restart is needed for the existing AuthorizedKeysFile location.
