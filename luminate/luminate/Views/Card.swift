//
//  Card.swift
//  luminate
//
//  Created by Prayash Thapa on 10/7/17.
//  Copyright © 2017 com.reality.af. All rights reserved.
//

import LBTAComponents

class Card: DatasourceCell {
    
    let imageView: UIImageView = {
        let view = UIImageView()
        view.image = UIImage(named: "card1")
        view.layer.cornerRadius = 0
        view.clipsToBounds = true
        return view
    }()
    
    override func setupViews() {
        super.setupViews()
        backgroundColor = .white
        
        addSubview(imageView)
        
        imageView.anchor(self.topAnchor, left: self.leftAnchor, bottom: nil, right: self.rightAnchor, topConstant: 12, leftConstant: 12, bottomConstant: 0, rightConstant: 12, widthConstant: 0, heightConstant: 450)
    }
}
