const path = require('path');

module.exports = {
  entry: './src/App.js',
  mode: process.env.NODE_ENV || "development",
  resolve: {
    extensions: ['*', '.js', '.jsx'],
  },
  output: {
    path: path.resolve(__dirname, '../static'),
    filename: 'bundle.js',
    publicPath: 'static/',
  },
  module: {
    rules: [
      {
        test: /\.js$/,
        exclude: /node_modules/,
        loader: 'babel-loader',
      },
      {
        test: /\.(woff|woff2)$/,
        use: {
          loader: 'file-loader',
          options: {
            name: '[name].[ext]',
            outputPath: 'webfonts/'
          }
        },
      },
      {
        test: /\.(sass|less|css)$/,
        use: ['style-loader', 'css-loader', 'less-loader']
      }
    ]
  },
  plugins: [
    {
       apply: (compiler) => {
         compiler.hooks.done.tap('DonePlugin', (stats) => {
           setTimeout(() => {
             process.exit(0)
           })
         });
       }
    }
  ]
};
