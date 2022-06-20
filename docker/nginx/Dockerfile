FROM nginx:latest

RUN rm /etc/nginx/conf.d/default.conf
ADD docker/nginx/etc/nginx/nginx.conf /etc/nginx/nginx.conf
RUN mkdir /etc/nginx/sites-enabled
ADD docker/nginx/etc/nginx/sites-enabled /etc/nginx/sites-enabled