from mars.lib.filesystem._oss_lib.common import convert_oss_path
import mars.dataframe as md
from mars.session import new_session


def main():
    session = new_session(default=True)
    
    # Replace with corresponding OSS information.
    access_key_id = 'your_access_key_id'
    access_key_secret = 'your_access_key_secret'
    end_point = 'your_endpoint'
    file_path = f"oss://bucket/test.csv"
    
    auth_path = convert_oss_path(file_path, access_key_id, access_key_secret, end_point)
    df = md.read_csv(auth_path).execute()
    print(df.shape)
    
 
if __name__ == "__main__":
    main()
