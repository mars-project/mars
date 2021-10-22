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

import PropTypes from 'prop-types';

export const nodeType = PropTypes.arrayOf(
  PropTypes.shape({
    id: PropTypes.string.isRequired,
    name: PropTypes.string,
  })
);

export const nodesStatusType = PropTypes.shape({
  id: PropTypes.shape({
    status: PropTypes.number.isRequired,
    progress: PropTypes.number.isRequired,
    subtaskCount: PropTypes.number,
    properties: PropTypes.shape({
      id: PropTypes.string,
    }),
  })
});

export const dependencyType = PropTypes.arrayOf(
  PropTypes.shape({
    fromNodeId: PropTypes.string.isRequired,
    toNodeId: PropTypes.string.isRequired,
    linkType: PropTypes.number,
  })
);

export const dagType = PropTypes.shape({
  margin: PropTypes.oneOfType([
    PropTypes.number,
    PropTypes.string,
  ]),
  padding: PropTypes.oneOfType([
    PropTypes.number,
    PropTypes.string,
  ]),
  width: PropTypes.oneOfType([
    PropTypes.number,
    PropTypes.string,
  ]),
  height: PropTypes.oneOfType([
    PropTypes.number,
    PropTypes.string,
  ]),
});
