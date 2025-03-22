#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

static AVCodec *codec = NULL;
static AVCodecContext *encoder = NULL;
static int pts = 0;

void init_libav(const int width, const int height){
    codec = avcodec_find_encoder(AV_CODEC_ID_H264);

    encoder = avcodec_alloc_context3(codec);

    encoder->bit_rate = 10 * 1000 * 10000;
    encoder->width = width;
    encoder->height = height;
    encoder->time_base = (AVRational) {1,30};
    encoder->gop_size = 30;
    encoder->max_b_frames = 1;
    encoder->pix_fmt = AV_PIX_FMT_RGB24;

    // av_opt_set(encoder->av_codec_context->priv_data, "preset", "ultrafast", 0);

    avcodec_open2(encoder, codec, NULL);
}


int send_frame(void *data, const int source_width, const int source_height){
    AVFrame* raw_frame = av_frame_alloc();
    raw_frame->data[0] = (uint8_t *) data;
    raw_frame->format = AV_PIX_FMT_RGB24;
    raw_frame->width  = source_width; 
    raw_frame->height = source_height;

    raw_frame->pts = pts++;
    avcodec_send_frame(encoder, raw_frame);

    av_freep(&raw_frame->data[0]);
    av_frame_free(&raw_frame);

    AVPacket encoded_frame; 
    int got_output = avcodec_receive_packet(encoder, &encoded_frame);

    if(got_output == 0) {
        fprintf(stderr,"Encoding went well!\n");
    }else{
        fprintf(stderr,"Encoding didn't go well!\n");
    }

    av_frame_unref(raw_frame);
}
