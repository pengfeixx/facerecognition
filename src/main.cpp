// SPDX-FileCopyrightText: 2020 - 2022 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "cmdline.h"
#include "src/interface/face.h"

#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;

void initCMD(int argc, char** argv)
{
    cmdline::parser a;
    a.add<string>("host", 'h', "host name", true, "");
    a.add<int>("port", 'p', "port number", false, 80, cmdline::range(1, 65535));
    a.add<string>("type", 't', "protocol type", false, "http", cmdline::oneof<string>("http", "https", "ssh", "ftp"));
    a.add("gzip", '\0', "gzip when transfer");

    a.parse_check(argc, argv);

    cout << a.get<string>("type") << "://"
         << a.get<string>("host") << ":"
         << a.get<int>("port") << endl;

    if (a.exist("gzip")) cout << "gzip" << endl;
}

int main(int argc, char** argv)
{
    // initCMD(argc, argv);

    Face face;
    const cv::Mat mat = cv::imread("/home/uos/work/face/face2/img/12.jpg");
    bool living = face.isliving(mat);
    printf("%d\n", living);

    return 0;
}
