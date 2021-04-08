#! /bin/bash
curl -v -X POST \
  'https://wzg93rdi1b.execute-api.us-east-1.amazonaws.com/test/convergence/{proxy+}' \
  -H 'content-type: application/json' \
  -d '{ "positive": ["dog"], "negative":["animal"] }'