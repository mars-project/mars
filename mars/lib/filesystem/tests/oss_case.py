# Copyright 1999-2021 Alibaba Group Holding Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import mars.dataframe as md
from mars.lib.filesystem.oss import build_oss_path
from mars.session import new_session


def main():
    session = new_session(default=True)
    
    # Replace with corresponding OSS information.
    access_key_id = 'your_access_key_id'
    access_key_secret = 'your_access_key_secret'
    end_point = 'your_endpoint'
    file_path = f"oss://bucket/test.csv"
    
    auth_path = build_oss_path(file_path, access_key_id, access_key_secret, end_point)
    df = md.read_csv(auth_path).execute()
    print(df.shape)
    
 
if __name__ == "__main__":
    main()
