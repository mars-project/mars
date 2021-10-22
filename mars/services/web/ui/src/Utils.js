/*
 * Copyright 1999-2021 Alibaba Group Holding Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

export function toReadableSize(size, trunc) {
  let res_size = size;
  let size_unit = '';

  if (size === null) {
    return 'NA';
  }
  if (size >= 1024 && size < 1024 ** 2) {
    res_size = size / 1024.0;
    size_unit = 'K';
  } else if (size >= 1024 ** 2 && size < 1024 ** 3) {
    res_size = size / 1024.0 ** 2;
    size_unit = 'M';
  } else if (size >= 1024 ** 3 && size < 1024 ** 4) {
    res_size = size / 1024.0 ** 3;
    size_unit = 'G';
  } else if (size >= 1024 ** 4) {
    res_size = size / 1024.0 ** 4;
    size_unit = 'T';
  }

  if (trunc === undefined) {
    return res_size.toFixed(2) + size_unit;
  }
  return Math.floor(res_size) + size_unit;
}

export function formatTime(time) {
  const date = new Date(time * 1000);
  const formatDigits = (n, d) => (`0${n}`).slice(-d);

  return `${date.getFullYear()
  }-${formatDigits(date.getMonth() + 1, 2)
  }-${formatDigits(date.getDate(), 2)
  } ${formatDigits(date.getHours(), 2)
  }:${formatDigits(date.getMinutes(), 2)
  }:${formatDigits(date.getSeconds(), 2)
  }.${formatDigits(date.getMilliseconds(), 3)}`;
}

export function getTaskStatusText(statusCode) {
  const mapping = {
    0: 'pending',
    1: 'running',
    2: 'terminated',
  };
  return mapping[statusCode];
}

export function getNodeStatusText(statusCode) {
  const mapping = {
    [-1]: 'stopped',
    0: 'starting',
    1: 'ready',
    2: 'degenerated',
    3: 'stopping',
  };
  return mapping[statusCode];
}
