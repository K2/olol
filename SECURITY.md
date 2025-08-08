# Security Policy

Security, sorry right now this is for way back end processing :).   Well there is some code to handle if you are accessing you're
infrance server over an ssh port forward, the discovery protocol should still work despite (at least works for me ;) where I have
the proxy locally and infrance server port forwarded to a remtoe server wher it sends the discovery packet with a differnt IP than 
the service port (which would be localhost) this tunneling sometimes causes a bit of a issue. 

That said, the assumption is trusted netowrking and very little has been done to secure anything, the discovery protocol can
easially blow up the router and so forth.

In the future if people request it, shal be done.

## Supported Versions

Use this section to tell people about which versions of your project are
currently being supported with security updates.

| Version | Supported          |
| ------- | ------------------ |
| 5.1.x   | :white_check_mark: |
| 5.0.x   | :x:                |
| 4.0.x   | :white_check_mark: |
| < 4.0   | :x:                |

## Reporting a Vulnerability

Use this section to tell people how to report a vulnerability.

Tell them where to go, how often they can expect to get an update on a
reported vulnerability, what to expect if the vulnerability is accepted or
declined, etc.
